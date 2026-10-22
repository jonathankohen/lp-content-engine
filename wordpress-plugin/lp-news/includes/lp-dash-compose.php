<?php

/**
 * The Compose tab: one brief plus an optional image, three social drafts out.
 *
 * WHY THIS IS NOT PART OF THE ASSISTANT TAB. The assistant's safety story is
 * its tool list: there is no tool that creates, publishes or sends, and that
 * sentence has to stay true (see the header of lp-dash-claude.php). Composing
 * is a different act, so it gets its own tab, its own AJAX endpoints and its
 * own nonce, and none of it is reachable from the chat loop.
 *
 * What it still will NOT do is approve anything. Every draft it creates lands
 * in Buffer as a draft, and a person clicks Approve or Send now on the Social
 * queue tab. That split is the client's (2026-08-19) and this tab keeps it.
 *
 * The three-stage flow, deliberately not one button:
 *   1. Upload  - the file goes into the WordPress media library and we keep the
 *                attachment id. Buffer fetches assets by URL, so the media
 *                library IS the hosting; there is nowhere else to put it.
 *   2. Draft   - one Anthropic call, no tools, returns JSON with three keys.
 *                The image is sent along so the copy can describe what is
 *                actually in the photo.
 *   3. Create  - the staff member edits all three, unticks any they do not
 *                want, and only then are the Buffer drafts written.
 */

if (! defined('ABSPATH')) {
	exit;
}

/**
 * Anthropic caps a single image at roughly 5MB base64, and a 12MP phone photo
 * blows straight past that. The 'large' intermediate size WordPress already
 * generated is well inside it and is plenty to write copy from.
 */
define('LP_DASH_VISION_MAX_BYTES', 4 * 1024 * 1024);

function lp_dash_compose_services()
{
	return array('linkedin' => 'LinkedIn', 'instagram' => 'Instagram', 'facebook' => 'Facebook');
}

/* -------------------------------------------------------------------------- *
 *  Attachments
 * -------------------------------------------------------------------------- */

/**
 * Normalise one attachment id into what both Buffer and Claude need from it.
 * Returns array('error') if the id is not an image or mp4 we can use.
 */
function lp_dash_compose_attachment($attachment_id)
{
	$attachment_id = (int) $attachment_id;
	$post = get_post($attachment_id);
	if (! $post || $post->post_type !== 'attachment') {
		return array('error' => 'That upload is no longer in the media library.');
	}

	$mime = (string) get_post_mime_type($attachment_id);
	$url  = wp_get_attachment_url($attachment_id);
	if (! $url) {
		return array('error' => 'That upload has no URL, so Buffer could not fetch it.');
	}

	$is_video = (strpos($mime, 'video/') === 0);
	if (! $is_video && strpos($mime, 'image/') !== 0) {
		return array('error' => 'Only images and mp4 video can be attached to a social post.');
	}

	return array(
		'id'       => $attachment_id,
		'url'      => $url,
		'mime'     => $mime,
		'is_video' => $is_video,
		'name'     => get_the_title($attachment_id),
		'thumb'    => $is_video ? '' : (string) wp_get_attachment_image_url($attachment_id, 'medium'),
	);
}

/**
 * An image as an Anthropic image content block, or null.
 *
 * Base64 rather than a source URL on purpose: Anthropic would have to reach the
 * media library itself, which is false on a staging or password-protected site
 * and would fail as a confusing API error rather than as "no image". Reading
 * the bytes here works wherever the plugin runs.
 *
 * Video is never sent. The Messages API takes images, not mp4, and pretending
 * otherwise would 400 the whole compose.
 */
function lp_dash_compose_image_block($attachment)
{
	if (empty($attachment) || ! empty($attachment['is_video'])) {
		return null;
	}

	// Prefer the already-generated 'large' file; fall back to the original for
	// an image too small to have intermediates.
	$path = '';
	$size = image_get_intermediate_size($attachment['id'], 'large');
	if (is_array($size) && ! empty($size['path'])) {
		$uploads = wp_get_upload_dir();
		$path    = trailingslashit($uploads['basedir']) . $size['path'];
	}
	if ($path === '' || ! file_exists($path)) {
		$path = (string) get_attached_file($attachment['id']);
	}
	if ($path === '' || ! file_exists($path)) {
		return null;
	}
	if (filesize($path) > LP_DASH_VISION_MAX_BYTES) {
		return null; // Too big to send; the brief alone still works.
	}

	$bytes = @file_get_contents($path);
	if ($bytes === false || $bytes === '') {
		return null;
	}

	// Trust the bytes, not the stored mime, same rule as lp_news_store_media().
	// There is deliberately NO fallback to $attachment['mime'] here: bytes that
	// GD cannot read are bytes Anthropic cannot read either, and labelling them
	// image/jpeg on the way out turns "no image" into a 400 on the whole call.
	$info = @getimagesizefromstring($bytes);
	$mime = (is_array($info) && ! empty($info['mime'])) ? (string) $info['mime'] : '';
	if (! in_array($mime, array('image/jpeg', 'image/png', 'image/gif', 'image/webp'), true)) {
		return null;
	}

	return array(
		'type'   => 'image',
		'source' => array(
			'type'       => 'base64',
			'media_type' => $mime,
			'data'       => base64_encode($bytes),
		),
	);
}

/* -------------------------------------------------------------------------- *
 *  Drafting
 * -------------------------------------------------------------------------- */

/**
 * The compose instructions, appended to the shared system prompt.
 *
 * The house rules (no dashes, the Concert of Kings rule, the IG and LinkedIn
 * CTAs) live in lp_dash_system_prompt() and are NOT restated here: two copies
 * of a style rule is how one of them goes stale.
 */
function lp_dash_compose_system_prompt()
{
	return lp_dash_system_prompt() . "\n" . implode("\n", array(
		'',
		'RIGHT NOW YOU ARE COMPOSING, NOT EDITING',
		'You are writing three brand new social posts from a brief: one for LinkedIn, one for Instagram, one for Facebook.',
		'These become DRAFTS. A person reads all three, edits them, and decides whether they ever go out.',
		'Write the same story three times, in each channel\'s own voice. Do not paste one text into all three.',
		'LinkedIn: professional, aimed at talent buyers and venues, a short paragraph or two.',
		'Instagram: warmer and shorter, the fixed link-in-bio line, at most 5 hashtags, no links or email addresses.',
		'Facebook: conversational, a little longer than Instagram is fine.',
		'If an image is attached, write about what is actually in it. Do not describe a photo you were not shown.',
		'If the brief is too thin to write from, say so in "notes" and still write the best three you can.',
		'',
		'OUTPUT FORMAT',
		'Reply with a single JSON object and nothing else. No preamble, no code fence.',
		'{"linkedin": "...", "instagram": "...", "facebook": "...", "notes": "..."}',
		'"notes" is one short line for the person, or an empty string. It is never part of a post.',
	));
}

/**
 * Pull the JSON object out of a model reply.
 *
 * Belt and braces on top of the format instruction: a stray "Here you go:" or a
 * ```json fence would otherwise turn a perfectly good draft into an error.
 */
function lp_dash_compose_parse($text)
{
	$text = trim((string) $text);
	if ($text === '') {
		return array('error' => 'Claude returned an empty reply.');
	}

	$decoded = json_decode($text, true);
	if (! is_array($decoded)) {
		$start = strpos($text, '{');
		$end   = strrpos($text, '}');
		if ($start !== false && $end !== false && $end > $start) {
			$decoded = json_decode(substr($text, $start, $end - $start + 1), true);
		}
	}
	if (! is_array($decoded)) {
		return array('error' => 'Claude did not return usable JSON. Try again, or reword the brief.');
	}

	$out = array('notes' => isset($decoded['notes']) ? trim((string) $decoded['notes']) : '');
	foreach (array_keys(lp_dash_compose_services()) as $service) {
		$out[$service] = isset($decoded[$service]) ? trim((string) $decoded[$service]) : '';
	}
	if ($out['linkedin'] === '' && $out['instagram'] === '' && $out['facebook'] === '') {
		return array('error' => 'Claude returned no copy for any channel. Try again.');
	}
	return $out;
}

/**
 * One shot, no tools: brief (plus image) in, three drafts out.
 */
function lp_dash_compose_generate($brief, $attachment = null, $extra = '')
{
	$key = lp_dash_anthropic_key();
	if ($key === '') {
		return array('error' => 'No Anthropic API key configured. Add one on the Settings tab.');
	}

	$content = array();
	$image   = lp_dash_compose_image_block($attachment);
	if ($image) {
		// Image first: the API's own guidance, and it reads better when the
		// text that follows can refer to "the image above".
		$content[] = $image;
	}

	$prompt = 'Brief from the team:' . "\n\n" . trim((string) $brief);
	if ($image) {
		$prompt .= "\n\n" . 'The attached image will go out with all three posts.';
	} elseif ($attachment && ! empty($attachment['is_video'])) {
		$prompt .= "\n\n" . 'A video clip will go out with all three posts. You cannot see it, so write from the brief and do not describe the footage.';
	}
	if (trim((string) $extra) !== '') {
		$prompt .= "\n\n" . 'Extra direction: ' . trim((string) $extra);
	}
	$content[] = array('type' => 'text', 'text' => $prompt);

	$body = array(
		'model'      => lp_dash_model(),
		'max_tokens' => LP_DASH_MAX_TOKENS,
		'system'     => lp_dash_compose_system_prompt(),
		'messages'   => array(array('role' => 'user', 'content' => $content)),
	);

	$resp = wp_remote_post(LP_DASH_ANTHROPIC_URL, array(
		'timeout' => 120,
		'headers' => array(
			'Content-Type'      => 'application/json',
			'x-api-key'         => $key,
			'anthropic-version' => LP_DASH_ANTHROPIC_VERSION,
		),
		'body'    => wp_json_encode($body),
	));

	if (is_wp_error($resp)) {
		return array('error' => 'Could not reach Claude: ' . $resp->get_error_message());
	}

	$code    = (int) wp_remote_retrieve_response_code($resp);
	$decoded = json_decode(wp_remote_retrieve_body($resp), true);

	if ($code === 401 || $code === 403) {
		return array('error' => 'Claude rejected the API key. Check it on the Settings tab.');
	}
	if ($code === 429) {
		return array('error' => 'Claude is rate limited right now. Try again in a moment.');
	}
	if ($code < 200 || $code >= 300) {
		$msg = isset($decoded['error']['message']) ? $decoded['error']['message'] : ('HTTP ' . $code);
		return array('error' => 'Claude error: ' . $msg);
	}
	if (! is_array($decoded)) {
		return array('error' => 'Claude returned an unreadable response.');
	}
	if (isset($decoded['stop_reason']) && $decoded['stop_reason'] === 'refusal') {
		return array('error' => 'Claude declined to write that one.');
	}

	$text = '';
	foreach ((isset($decoded['content']) && is_array($decoded['content']) ? $decoded['content'] : array()) as $block) {
		if (isset($block['type']) && $block['type'] === 'text') {
			$text .= isset($block['text']) ? $block['text'] : '';
		}
	}
	return lp_dash_compose_parse($text);
}

/* -------------------------------------------------------------------------- *
 *  AJAX
 * -------------------------------------------------------------------------- */

add_action('wp_ajax_lp_dash_compose_upload', 'lp_dash_ajax_compose_upload');

/**
 * Take the file into the media library and hand back its public URL.
 *
 * media_handle_upload() rather than a hand-rolled writer: it applies the site's
 * own allowed mime types and filename sanitising, and it generates the
 * intermediate sizes that the vision path then reads instead of the original.
 */
function lp_dash_ajax_compose_upload()
{
	if (! current_user_can(lp_dash_capability()) || ! current_user_can('upload_files')) {
		wp_send_json_error(array('message' => 'You do not have permission to upload files.'), 403);
	}
	check_ajax_referer('lp_dash_compose', 'nonce');

	if (empty($_FILES['file']) || ! isset($_FILES['file']['tmp_name'])) {
		wp_send_json_error(array('message' => 'No file arrived. It may be larger than this server accepts.'), 400);
	}

	require_once ABSPATH . 'wp-admin/includes/file.php';
	require_once ABSPATH . 'wp-admin/includes/media.php';
	require_once ABSPATH . 'wp-admin/includes/image.php';

	$attachment_id = media_handle_upload('file', 0);
	if (is_wp_error($attachment_id)) {
		wp_send_json_error(array('message' => $attachment_id->get_error_message()), 400);
	}

	$attachment = lp_dash_compose_attachment($attachment_id);
	if (isset($attachment['error'])) {
		// Not something to post, so do not leave it lying in the library.
		wp_delete_attachment((int) $attachment_id, true);
		wp_send_json_error(array('message' => $attachment['error']), 400);
	}

	wp_send_json_success($attachment);
}

add_action('wp_ajax_lp_dash_compose_draft', 'lp_dash_ajax_compose_draft');

function lp_dash_ajax_compose_draft()
{
	if (! current_user_can(lp_dash_capability())) {
		wp_send_json_error(array('message' => 'You do not have permission to use this.'), 403);
	}
	check_ajax_referer('lp_dash_compose', 'nonce');

	$brief = isset($_POST['brief']) ? sanitize_textarea_field(wp_unslash($_POST['brief'])) : '';
	$extra = isset($_POST['extra']) ? sanitize_textarea_field(wp_unslash($_POST['extra'])) : '';
	if (trim($brief) === '') {
		wp_send_json_error(array('message' => 'Say what the post is about first.'), 400);
	}
	if (strlen($brief) > 4000) {
		wp_send_json_error(array('message' => 'That brief is too long. Keep it under 4000 characters.'), 400);
	}

	$attachment = null;
	if (! empty($_POST['attachment_id'])) {
		$attachment = lp_dash_compose_attachment((int) $_POST['attachment_id']);
		if (isset($attachment['error'])) {
			wp_send_json_error(array('message' => $attachment['error']), 400);
		}
	}

	$result = lp_dash_compose_generate($brief, $attachment, $extra);
	if (isset($result['error'])) {
		wp_send_json_error(array('message' => $result['error']), 200);
	}
	wp_send_json_success($result);
}

add_action('wp_ajax_lp_dash_compose_create', 'lp_dash_ajax_compose_create');

/**
 * Write the drafts to Buffer, one channel at a time.
 *
 * Each channel is reported on its own. Buffer has no transaction across three
 * createPost calls, so if the third fails the first two really did get created,
 * and saying "it failed" would send someone looking for posts that exist.
 */
function lp_dash_ajax_compose_create()
{
	if (! current_user_can(lp_dash_capability())) {
		wp_send_json_error(array('message' => 'You do not have permission to use this.'), 403);
	}
	check_ajax_referer('lp_dash_compose', 'nonce');

	$attachment = null;
	if (! empty($_POST['attachment_id'])) {
		$attachment = lp_dash_compose_attachment((int) $_POST['attachment_id']);
		if (isset($attachment['error'])) {
			wp_send_json_error(array('message' => $attachment['error']), 400);
		}
	}

	$media = array('images' => array(), 'video' => '', 'thumbnail' => '');
	if ($attachment) {
		if ($attachment['is_video']) {
			$media['video'] = $attachment['url'];
		} else {
			$media['images'] = array($attachment['url']);
		}
	}

	$when = isset($_POST['when']) ? sanitize_text_field(wp_unslash($_POST['when'])) : '';
	$when = trim($when) !== '' ? $when : null;

	$results = array();
	$any     = false;

	foreach (lp_dash_compose_services() as $service => $label) {
		if (empty($_POST[$service . '_on'])) {
			continue;
		}
		$text = isset($_POST[$service]) ? sanitize_textarea_field(wp_unslash($_POST[$service])) : '';
		if (trim($text) === '') {
			$results[] = array('service' => $service, 'ok' => false, 'message' => $label . ': nothing to post, the box is empty.');
			continue;
		}

		$res = lp_dash_buffer_create($service, $text, $media, $when);
		$ok  = ! is_array($res);
		$any = $any || $ok;

		lp_dash_audit(
			'human:compose_draft',
			array('service' => $service, 'attachment' => $attachment ? $attachment['id'] : 0, 'when' => $when, 'chars' => strlen($text)),
			$ok,
			$ok ? '' : $res['error']
		);

		$results[] = array(
			'service' => $service,
			'ok'      => $ok,
			'message' => $ok ? ($label . ': draft created.') : ($label . ': ' . $res['error']),
		);
	}

	if (empty($results)) {
		wp_send_json_error(array('message' => 'Tick at least one channel.'), 400);
	}
	wp_send_json_success(array('results' => $results, 'any' => $any));
}

/* -------------------------------------------------------------------------- *
 *  The tab
 * -------------------------------------------------------------------------- */

function lp_dash_render_compose()
{
	if (lp_dash_anthropic_key() === '') {
		echo '<div class="notice notice-warning"><p>No Anthropic API key yet. Add one on the '
			. '<a href="' . esc_url(lp_dash_tab_url('settings')) . '">Settings</a> tab to write drafts here.</p></div>';
		return;
	}
	if (lp_dash_buffer_token() === '') {
		echo '<div class="notice notice-warning"><p>No Buffer token yet. Add one on the '
			. '<a href="' . esc_url(lp_dash_tab_url('settings')) . '">Settings</a> tab, or the drafts will have nowhere to go.</p></div>';
		return;
	}

	$channels = lp_dash_buffer_channels();
	if (isset($channels['error'])) {
		echo '<div class="notice notice-error"><p>' . esc_html($channels['error']) . '</p></div>';
		return;
	}

	$nonce    = wp_create_nonce('lp_dash_compose');
	$services = lp_dash_compose_services();
	$default  = gmdate('Y-m-d\TH:i', time() + 86400);
?>
	<p style="max-width:820px">
		Describe the post once and attach a photo or a clip. Claude writes it three ways, one per channel, and you
		edit all three before anything is created. Everything lands in Buffer as a <strong>draft</strong>: sending is
		still the Approve and Send now buttons on the
		<a href="<?php echo esc_url(lp_dash_tab_url('social')); ?>">Social queue</a> tab, for a person to click.
	</p>
	<p style="max-width:820px;color:#666">
		The file goes into this site's media library, because Buffer fetches attachments by URL. That means it has to
		be reachable from the public internet: on a staging or password-protected site the drafts will be created but
		the image will not attach.
	</p>

	<h2>1. What is the post about?</h2>
	<textarea id="lpc-brief" rows="4" style="width:100%;max-width:860px"
		placeholder="e.g. The Concert of Kings plays the Paramount in Huntington on 14 November, doors 7pm, tickets on sale now."></textarea>

	<h2 style="margin-top:24px">2. Image or video <span style="color:#888;font-weight:400">(optional, but Instagram needs one)</span></h2>
	<p>
		<input type="file" id="lpc-file" accept="image/*,video/mp4">
		<span id="lpc-file-status" style="margin-left:10px;color:#666"></span>
	</p>
	<div id="lpc-preview" style="margin:8px 0 4px;display:none">
		<img id="lpc-preview-img" alt="" style="max-width:260px;height:auto;border:1px solid #dcdcde">
		<div id="lpc-preview-video" style="display:none;color:#666">Video attached. Claude cannot see it, so the copy comes from your brief alone.</div>
		<p><button type="button" class="button" id="lpc-file-clear">Remove</button></p>
	</div>
	<input type="hidden" id="lpc-attachment" value="">

	<h2 style="margin-top:24px">3. Write the drafts</h2>
	<p>
		<input type="text" id="lpc-extra" class="regular-text" style="max-width:520px"
			placeholder="Any extra steer? e.g. lead with the venue, keep it short">
		<button class="button button-primary" id="lpc-draft" style="margin-left:6px">Write three drafts</button>
		<span id="lpc-draft-status" style="margin-left:10px;color:#666"></span>
	</p>
	<p id="lpc-notes" style="max-width:860px;color:#996800;display:none"></p>

	<div id="lpc-results" style="display:none">
		<h2 style="margin-top:24px">4. Edit, then create</h2>
		<div style="display:flex;gap:16px;flex-wrap:wrap;max-width:1400px">
			<?php foreach ($services as $service => $label) : ?>
				<div style="flex:1 1 320px;min-width:300px">
					<p style="margin:0 0 6px">
						<label>
							<input type="checkbox" class="lpc-on" data-service="<?php echo esc_attr($service); ?>" checked>
							<strong><?php echo esc_html($label); ?></strong>
						</label>
						<?php if (isset($channels[$service])) : ?>
							<span style="color:#888;font-size:12px"><?php echo esc_html($channels[$service]['name']); ?></span>
						<?php else : ?>
							<span style="color:#d63638;font-size:12px">not connected in Buffer</span>
						<?php endif; ?>
						<span class="lpc-count" data-service="<?php echo esc_attr($service); ?>"
							style="float:right;color:#888;font-size:12px"></span>
					</p>
					<textarea class="lpc-copy" data-service="<?php echo esc_attr($service); ?>" rows="14"
						style="width:100%"<?php echo isset($channels[$service]) ? '' : ' disabled'; ?>></textarea>
				</div>
			<?php endforeach; ?>
		</div>

		<p style="margin-top:16px">
			<label for="lpc-when">Slot them for</label>
			<input type="datetime-local" id="lpc-when" value="<?php echo esc_attr($default); ?>" style="width:200px">
			<span style="color:#666;margin-left:8px">Only the draft's suggested time. It still needs approving to send.</span>
		</p>
		<p>
			<button class="button button-primary" id="lpc-create">Create the drafts in Buffer</button>
			<button class="button" id="lpc-clear">Start over</button>
			<span id="lpc-create-status" style="margin-left:10px;color:#666"></span>
		</p>
		<div id="lpc-outcome"></div>
	</div>

	<script>
		(function() {
			var nonce = <?php echo wp_json_encode($nonce); ?>,
				ajax = <?php echo wp_json_encode(admin_url('admin-ajax.php')); ?>,
				social = <?php echo wp_json_encode(lp_dash_tab_url('social')); ?>,
				services = <?php echo wp_json_encode(array_keys($services)); ?>,
				busy = false;

			function $(id) {
				return document.getElementById(id);
			}

			function each(sel, fn) {
				Array.prototype.forEach.call(document.querySelectorAll(sel), fn);
			}

			function box(service) {
				return document.querySelector('.lpc-copy[data-service="' + service + '"]');
			}

			function toggle(service) {
				return document.querySelector('.lpc-on[data-service="' + service + '"]');
			}

			// FormData rather than urlencoded, because the upload route needs it
			// and one transport for all three keeps the error handling in one place.
			function post(action, form, done) {
				form.append('action', action);
				form.append('nonce', nonce);
				fetch(ajax, {
						method: 'POST',
						credentials: 'same-origin',
						body: form
					})
					.then(function(r) {
						return r.json();
					})
					.then(done)
					.catch(function() {
						done({
							success: false,
							data: {
								message: 'The request failed. If it was a long one it may have timed out; check the Social queue before retrying.'
							}
						});
					});
			}

			function counts() {
				each('.lpc-count', function(el) {
					var t = box(el.dataset.service);
					el.textContent = t && t.value ? t.value.length + ' characters' : '';
				});
			}

			/* ---- upload ---- */

			$('lpc-file').addEventListener('change', function() {
				var f = this.files && this.files[0];
				if (!f) return;
				var form = new FormData();
				form.append('file', f);
				$('lpc-file-status').textContent = 'Uploading...';
				post('lp_dash_compose_upload', form, function(res) {
					if (res && res.success) {
						$('lpc-attachment').value = res.data.id;
						$('lpc-file-status').textContent = 'Attached: ' + res.data.name;
						$('lpc-preview').style.display = 'block';
						if (res.data.is_video) {
							$('lpc-preview-img').style.display = 'none';
							$('lpc-preview-video').style.display = 'block';
						} else {
							$('lpc-preview-img').src = res.data.thumb || res.data.url;
							$('lpc-preview-img').style.display = 'inline-block';
							$('lpc-preview-video').style.display = 'none';
						}
					} else {
						$('lpc-file-status').textContent = (res && res.data && res.data.message) || 'Upload failed.';
					}
				});
			});

			$('lpc-file-clear').addEventListener('click', function() {
				// The attachment stays in the media library on purpose: it may
				// already be on a draft, and deleting it would break that post.
				$('lpc-attachment').value = '';
				$('lpc-file').value = '';
				$('lpc-file-status').textContent = '';
				$('lpc-preview').style.display = 'none';
			});

			/* ---- draft ---- */

			$('lpc-draft').addEventListener('click', function() {
				if (busy) return;
				var brief = $('lpc-brief').value.trim();
				if (!brief) {
					$('lpc-draft-status').textContent = 'Say what the post is about first.';
					return;
				}
				busy = true;
				this.disabled = true;
				$('lpc-draft-status').textContent = 'Writing, this takes a few seconds...';

				var form = new FormData();
				form.append('brief', brief);
				form.append('extra', $('lpc-extra').value);
				form.append('attachment_id', $('lpc-attachment').value);

				post('lp_dash_compose_draft', form, function(res) {
					busy = false;
					$('lpc-draft').disabled = false;
					if (res && res.success) {
						$('lpc-draft-status').textContent = 'Drafted. Edit anything below before creating.';
						services.forEach(function(s) {
							var t = box(s);
							if (t && !t.disabled) t.value = res.data[s] || '';
							var on = toggle(s);
							if (on) on.checked = !!(res.data[s] && t && !t.disabled);
						});
						if (res.data.notes) {
							$('lpc-notes').textContent = 'Claude noted: ' + res.data.notes;
							$('lpc-notes').style.display = 'block';
						} else {
							$('lpc-notes').style.display = 'none';
						}
						$('lpc-results').style.display = 'block';
						$('lpc-outcome').innerHTML = '';
						counts();
					} else {
						$('lpc-draft-status').textContent = (res && res.data && res.data.message) || 'Something went wrong.';
					}
				});
			});

			/* ---- create ---- */

			$('lpc-create').addEventListener('click', function() {
				if (busy) return;
				var picked = services.filter(function(s) {
					var on = toggle(s);
					return on && on.checked && box(s) && box(s).value.trim();
				});
				if (!picked.length) {
					$('lpc-create-status').textContent = 'Nothing ticked with copy in it.';
					return;
				}
				if (!confirm('Create ' + picked.length + ' Buffer draft' + (picked.length > 1 ? 's' : '') + '? They will not send until someone approves them.')) {
					return;
				}
				busy = true;
				this.disabled = true;
				$('lpc-create-status').textContent = 'Creating...';

				var form = new FormData();
				form.append('attachment_id', $('lpc-attachment').value);
				form.append('when', $('lpc-when').value);
				picked.forEach(function(s) {
					form.append(s + '_on', '1');
					form.append(s, box(s).value);
				});

				post('lp_dash_compose_create', form, function(res) {
					busy = false;
					$('lpc-create').disabled = false;
					$('lpc-create-status').textContent = '';
					var out = $('lpc-outcome');
					out.innerHTML = '';
					if (!res || !res.success) {
						out.innerHTML = '<div class="notice notice-error inline"><p></p></div>';
						out.querySelector('p').textContent = (res && res.data && res.data.message) || 'Something went wrong.';
						return;
					}
					// Per channel, because Buffer has no transaction across the
					// three calls and a partial success is a real outcome.
					res.data.results.forEach(function(r) {
						var d = document.createElement('div');
						d.className = 'notice inline notice-' + (r.ok ? 'success' : 'error');
						var p = document.createElement('p');
						p.textContent = r.message;
						d.appendChild(p);
						out.appendChild(d);
					});
					if (res.data.any) {
						var d = document.createElement('p');
						d.innerHTML = '<a class="button button-primary" href="' + social + '&refresh=1">See them on the Social queue</a>';
						out.appendChild(d);
					}
				});
			});

			$('lpc-clear').addEventListener('click', function() {
				services.forEach(function(s) {
					var t = box(s);
					if (t) t.value = '';
				});
				$('lpc-brief').value = '';
				$('lpc-extra').value = '';
				$('lpc-results').style.display = 'none';
				$('lpc-notes').style.display = 'none';
				$('lpc-draft-status').textContent = '';
				$('lpc-file-clear').click();
			});

			each('.lpc-copy', function(el) {
				el.addEventListener('input', counts);
			});
		})();
	</script>
<?php
}
