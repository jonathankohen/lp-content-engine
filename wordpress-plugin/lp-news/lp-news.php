<?php

/**
 * Plugin Name:       LP News
 * Description:        Creates standard WordPress news posts (draft) from the LP Content Engine weekly run. Each post gets a featured image, body, category, and a red "Read more" button linking to the source.
 * Version:           1.0.0
 * Author:            Love Productions
 * License:           GPL-2.0-or-later
 * Requires at least: 5.8
 * Requires PHP:      7.4
 *
 * Integration overview
 * --------------------
 * 1. The Python job (lp-content-engine, lp/wordpress.py::publish_news_posts) POSTs
 *        { "dry_run"?: bool, "limit"?: int, "posts": [ { key, title, body,
 *          categories[], button_url, button_label, image_url } ] }
 *    to  /wp-json/lp-news/v1/publish-news  with header  X-LP-News-Secret: <secret>.
 * 2. This plugin validates the secret and, for each post not already created
 *    (deduped by its `key` stored as post meta `_lp_news_key`), inserts a DRAFT
 *    `post`: title = headline, body = paragraph blocks + a red "Read more" button,
 *    built-in `category` terms assigned by name (restricted to the allowlist
 *    below, created if missing), and the featured image sideloaded from image_url.
 *
 * This plugin is intentionally separate from the Tour Calendar plugin: that one
 * owns `event` posts + the calendar shortcode; this one owns standard news posts.
 * They share no state.
 */

if (! defined('ABSPATH')) {
	exit; // No direct access.
}

define('LP_NEWS_VERSION', '1.0.0');
define('LP_NEWS_OPTION_SECRET', 'lp_news_secret');
// ID of a post configured with the desired theme "Page settings"
// (Header/Footer/Sidebar/Layout/etc.); its meta is copied onto every new post.
define('LP_NEWS_OPTION_TEMPLATE', 'lp_news_template_post_id');

/**
 * The only categories this endpoint will ever apply. Names in the payload that
 * are not on this list are dropped, so a typo or unexpected value can't spawn a
 * stray term. Names on the list are created if they don't yet exist.
 */
function lp_news_allowed_categories()
{
	return array(
		'Celebration',
		'Celebrity',
		'Theatre',
		'Tour',
		'Condolences',
		'Festival',
		'Interview',
		'Sold Out',
		'Tribute',
		'TV Show',
		'Uncategorized',
	);
}

/**
 * Resolve the shared secret.
 *
 * Prefer a constant in wp-config.php (define('LP_NEWS_SECRET', '...'); not
 * stored in the database, the most secure option). Otherwise fall back to an
 * auto-generated option created on activation.
 */
function lp_news_get_secret()
{
	if (defined('LP_NEWS_SECRET') && LP_NEWS_SECRET) {
		return (string) LP_NEWS_SECRET;
	}
	return (string) get_option(LP_NEWS_OPTION_SECRET, '');
}

/**
 * Generate a secret on activation if one is not already configured.
 */
function lp_news_activate()
{
	if (! defined('LP_NEWS_SECRET') && ! get_option(LP_NEWS_OPTION_SECRET)) {
		add_option(LP_NEWS_OPTION_SECRET, wp_generate_password(48, false, false));
	}
}
register_activation_hook(__FILE__, 'lp_news_activate');

/* -------------------------------------------------------------------------- *
 *  REST: publish-news endpoint
 * -------------------------------------------------------------------------- */

add_action('rest_api_init', function () {
	register_rest_route(
		'lp-news/v1',
		'/publish-news',
		array(
			'methods'             => 'POST',
			'callback'            => 'lp_news_rest_publish_news',
			'permission_callback' => 'lp_news_rest_permission',
		)
	);

	register_rest_route(
		'lp-news/v1',
		'/upload-media',
		array(
			'methods'             => 'POST',
			'callback'            => 'lp_news_rest_upload_media',
			'permission_callback' => 'lp_news_rest_permission',
		)
	);
});

/**
 * Accept raw image bytes and return a public media-library URL.
 *
 * publish-news sideloads a featured image from a URL, which is useless for the
 * branded cards the content engine renders locally: Buffer needs a public URL
 * for an image asset, so the bytes have to reach the site before any URL exists.
 * This endpoint closes that loop.
 *
 * Body: { key, filename, content_base64 }. `key` dedups: re-uploading the same
 * card returns the existing attachment rather than filling the media library
 * with duplicates on every re-run (card rendering is deterministic, so the same
 * card genuinely is the same file).
 */
function lp_news_rest_upload_media(WP_REST_Request $request)
{
	$body = $request->get_json_params();
	if (! is_array($body)) {
		return new WP_REST_Response(array('error' => 'Body must be a JSON object.'), 400);
	}

	$key      = isset($body['key']) ? sanitize_text_field($body['key']) : '';
	$filename = isset($body['filename']) ? sanitize_file_name($body['filename']) : '';
	$b64      = isset($body['content_base64']) ? (string) $body['content_base64'] : '';

	if ('' === $key || '' === $filename || '' === $b64) {
		return new WP_REST_Response(array('error' => 'key, filename and content_base64 are all required.'), 400);
	}

	// Only the formats this pipeline actually produces: cards from the renderer,
	// and mp4 clips cut from the agency's own Vimeo footage.
	$ext = strtolower(pathinfo($filename, PATHINFO_EXTENSION));
	if (! in_array($ext, array('png', 'jpg', 'jpeg', 'mp4'), true)) {
		return new WP_REST_Response(array('error' => 'Only png, jpg and mp4 uploads are accepted.'), 400);
	}
	$is_video = ('mp4' === $ext);

	$existing = get_posts(array(
		'post_type'      => 'attachment',
		'post_status'    => 'inherit',
		'posts_per_page' => 1,
		'fields'         => 'ids',
		'meta_key'       => '_lp_news_media_key',
		'meta_value'     => $key,
	));
	if (! empty($existing)) {
		return new WP_REST_Response(array(
			'id'      => (int) $existing[0],
			'url'     => wp_get_attachment_url((int) $existing[0]),
			'skipped' => true,
		), 200);
	}

	$bytes = base64_decode($b64, true);
	if (false === $bytes || '' === $bytes) {
		return new WP_REST_Response(array('error' => 'content_base64 is not valid base64.'), 400);
	}

	// Trust the bytes, not the caller's extension. Images can be checked
	// directly; for mp4 the ftyp box near the head of the file is the cheap
	// equivalent, since getimagesizefromstring says nothing about video.
	if ($is_video) {
		if (false === strpos(substr($bytes, 0, 32), 'ftyp')) {
			return new WP_REST_Response(array('error' => 'Payload is not a valid mp4.'), 400);
		}
		$mime = 'video/mp4';
	} else {
		$info = @getimagesizefromstring($bytes);
		if (false === $info || empty($info['mime']) || 0 !== strpos($info['mime'], 'image/')) {
			return new WP_REST_Response(array('error' => 'Payload is not a valid image.'), 400);
		}
		$mime = $info['mime'];
	}

	require_once ABSPATH . 'wp-admin/includes/file.php';
	require_once ABSPATH . 'wp-admin/includes/image.php';

	$upload = wp_upload_bits($filename, null, $bytes);
	if (! empty($upload['error'])) {
		return new WP_REST_Response(array('error' => $upload['error']), 500);
	}

	$att_id = wp_insert_attachment(array(
		'post_mime_type' => $mime,
		'post_title'     => sanitize_text_field(pathinfo($filename, PATHINFO_FILENAME)),
		'post_content'   => '',
		'post_status'    => 'inherit',
	), $upload['file']);

	if (is_wp_error($att_id) || ! $att_id) {
		@unlink($upload['file']);
		return new WP_REST_Response(array('error' => 'Could not create the attachment.'), 500);
	}

	wp_update_attachment_metadata($att_id, wp_generate_attachment_metadata($att_id, $upload['file']));
	update_post_meta($att_id, '_lp_news_media_key', $key);

	return new WP_REST_Response(array(
		'id'      => (int) $att_id,
		'url'     => wp_get_attachment_url($att_id),
		'skipped' => false,
	), 200);
}

/**
 * Constant-time secret check via the X-LP-News-Secret header.
 */
function lp_news_rest_permission(WP_REST_Request $request)
{
	$expected = lp_news_get_secret();
	$provided = (string) $request->get_header('x-lp-news-secret');

	if ('' === $expected) {
		return new WP_Error('lp_news_no_secret', 'News secret is not configured on the server.', array('status' => 500));
	}
	if (! hash_equals($expected, $provided)) {
		return new WP_Error('lp_news_forbidden', 'Invalid or missing X-LP-News-Secret header.', array('status' => 403));
	}
	return true;
}

/**
 * Create draft news posts from the payload.
 *
 * Body: { dry_run?: bool, limit?: int, posts: [ { key, title, body,
 * categories[], button_url, button_label, image_url } ] }. Each post is created
 * once, re-sending the same `key` is a no-op (reported as skipped). `title` is
 * required; everything else degrades gracefully.
 */
function lp_news_rest_publish_news(WP_REST_Request $request)
{
	$body = $request->get_json_params();

	if (! is_array($body) || ! isset($body['posts']) || ! is_array($body['posts'])) {
		return new WP_REST_Response(array('error' => 'Body must be an object with a "posts" array.'), 400);
	}

	// Image sideloading can outlast the default 30s execution cap. Lift it where
	// the host allows (no-op when disabled).
	if (function_exists('set_time_limit')) {
		@set_time_limit(0);
	}

	$dry_run = ! empty($body['dry_run']);
	$limit   = isset($body['limit']) ? max(0, (int) $body['limit']) : 0;

	$created      = array();
	$skipped      = array();
	$would_create = array();
	$errors       = array();
	$made         = 0;

	foreach ($body['posts'] as $post) {
		if (! is_array($post)) {
			continue;
		}
		$title = isset($post['title']) ? sanitize_text_field((string) $post['title']) : '';
		if ('' === $title) {
			continue;
		}
		// Dedup key: an explicit key wins, else fall back to the title. Stored on
		// the post as `_lp_news_key` so re-runs never duplicate the same story.
		$key = isset($post['key']) ? sanitize_text_field((string) $post['key']) : '';
		if ('' === $key) {
			$key = $title;
		}

		$existing = get_posts(array(
			'post_type'   => 'post',
			'post_status' => array('publish', 'future', 'draft', 'pending', 'private'),
			'numberposts' => 1,
			'fields'      => 'ids',
			'meta_key'    => '_lp_news_key',
			'meta_value'  => $key,
		));
		if (! empty($existing)) {
			$skipped[] = array('key' => $key, 'title' => $title, 'reason' => 'exists', 'id' => (int) $existing[0]);
			continue;
		}

		$body_text    = isset($post['body']) ? (string) $post['body'] : '';
		$button_url   = isset($post['button_url']) ? esc_url_raw((string) $post['button_url']) : '';
		$button_label = isset($post['button_label']) ? sanitize_text_field((string) $post['button_label']) : 'Read more';
		$image_url    = isset($post['image_url']) ? esc_url_raw((string) $post['image_url']) : '';
		$cats         = (isset($post['categories']) && is_array($post['categories'])) ? $post['categories'] : array();
		$cat_names    = lp_news_resolve_categories($cats);

		// Content: body paragraphs + the red "Read more" button (when a URL exists).
		$content = lp_news_text_to_blocks($body_text);
		if ('' !== $button_url) {
			$content = rtrim($content) . lp_news_button_html($button_url, $button_label);
		}

		if ($limit > 0 && $made >= $limit) {
			break;
		}

		$plan = array(
			'key'        => $key,
			'title'      => $title,
			'categories' => $cat_names,
			'button_url' => $button_url,
			'has_image'  => ('' !== $image_url),
		);

		if ($dry_run) {
			$would_create[] = $plan;
			$made++;
			continue;
		}

		$new_id = wp_insert_post(
			array(
				'post_type'    => 'post',
				'post_status'  => 'draft',
				'post_title'   => $title,
				'post_content' => $content,
			),
			true
		);
		if (is_wp_error($new_id)) {
			$errors[] = array('key' => $key, 'title' => $title, 'error' => $new_id->get_error_message());
			continue;
		}

		update_post_meta($new_id, '_lp_news_key', $key);

		// Stamp the theme's page-settings meta (Header/Footer/Sidebar/Layout/etc.)
		// copied from the configured template post, so LP News posts don't fall
		// back to the theme's global defaults.
		lp_news_copy_template_meta($new_id);

		$applied_cats = lp_news_assign_categories($new_id, $cat_names);

		$image_attached = false;
		if ('' !== $image_url) {
			$att_id = lp_news_sideload_url($image_url, $new_id);
			if ($att_id && ! is_wp_error($att_id)) {
				set_post_thumbnail($new_id, $att_id);
				$image_attached = true;
			}
		}

		$made++;
		$created[] = array(
			'id'             => (int) $new_id,
			'key'            => $key,
			'title'          => $title,
			'edit_link'      => get_edit_post_link($new_id, ''),
			'categories'     => $applied_cats,
			'button_url'     => $button_url,
			'image_attached' => $image_attached,
		);
	}

	return new WP_REST_Response(array(
		'dry_run'      => $dry_run,
		'created'      => $created,
		'skipped'      => $skipped,
		'would_create' => $would_create,
		'errors'       => $errors,
	), 200);
}

/* -------------------------------------------------------------------------- *
 *  Helpers
 * -------------------------------------------------------------------------- */

/**
 * Filter requested category names down to the allowlist (case-insensitive),
 * returning the canonical-cased names. Drops anything not on the list and
 * de-duplicates. Falls back to "Uncategorized" when nothing valid is left.
 */
function lp_news_resolve_categories($names)
{
	$allowed = lp_news_allowed_categories();
	$lookup  = array();
	foreach ($allowed as $a) {
		$lookup[strtolower($a)] = $a;
	}
	$out = array();
	foreach ((array) $names as $name) {
		$key = strtolower(trim((string) $name));
		if ('' === $key || ! isset($lookup[$key])) {
			continue;
		}
		$canonical = $lookup[$key];
		if (! in_array($canonical, $out, true)) {
			$out[] = $canonical;
		}
	}
	if (empty($out)) {
		$out[] = 'Uncategorized';
	}
	return $out;
}

/**
 * Assign built-in `category` terms (by name) to a post, creating any allowlisted
 * term that doesn't yet exist. Replaces existing terms. Returns names applied.
 * `$names` is assumed already filtered by lp_news_resolve_categories().
 */
function lp_news_assign_categories($post_id, $names)
{
	$term_ids = array();
	$applied  = array();
	foreach ((array) $names as $name) {
		$name = trim((string) $name);
		if ('' === $name) {
			continue;
		}
		$term = get_term_by('name', $name, 'category');
		if (! $term) {
			$new = wp_insert_term($name, 'category');
			if (! is_wp_error($new)) {
				$term_ids[] = (int) $new['term_id'];
				$applied[]  = $name;
			}
			continue;
		}
		if (! is_wp_error($term)) {
			$term_ids[] = (int) $term->term_id;
			$applied[]  = $term->name;
		}
	}
	if ($term_ids) {
		wp_set_object_terms($post_id, $term_ids, 'category', false);
	}
	return $applied;
}

/**
 * Cleanup pass for plain text before it becomes paragraph blocks: normalize line
 * endings/spaces, collapse blank runs, and unwrap hard-wrapped paragraphs while
 * preserving blank-line-delimited paragraph structure. Returns '' for empty input.
 */
function lp_news_clean_text($text)
{
	$text = (string) $text;
	$text = str_replace(array("\r\n", "\r"), "\n", $text);
	$text = str_replace(array("\xC2\xA0", "\xE2\x80\xAF"), ' ', $text); // NBSP, narrow NBSP
	$text = preg_replace('/[ \t]+\n/', "\n", $text);
	$text = preg_replace('/\n{3,}/', "\n\n", trim($text));

	$paragraphs = array();
	foreach (preg_split('/\n\n/', $text) as $para) {
		$para = trim($para);
		$para = preg_replace('/(\p{L})-\n[ \t]*(\p{L})/u', '$1-$2', $para);
		$para = preg_replace('/\s*\n\s*/', ' ', $para);
		$para = preg_replace('/[ \t]{2,}/', ' ', $para);
		if ('' !== trim($para)) {
			$paragraphs[] = $para;
		}
	}
	return implode("\n\n", $paragraphs);
}

/**
 * First-line indent prepended to every body paragraph after the first, to give
 * the article a "like a book" look. Em spaces (U+2003) are used because leading
 * ASCII whitespace/tabs collapse in HTML; roughly one tab's worth. Tunable, add
 * or remove &emsp; entities to widen or narrow the indent.
 */
define('LP_NEWS_INDENT', '&emsp;&emsp;');

/**
 * Convert plain text into Gutenberg blocks, formatted "like a book" so the post
 * renders with no device-dependent inter-paragraph gaps:
 *   - The narrative paragraphs become ONE paragraph block, separated by <br>
 *     (a hard line break, rendered consistently everywhere) instead of separate
 *     <p> blocks (whose theme margins are what vary by device). Every paragraph
 *     after the first is first-line indented (LP_NEWS_INDENT).
 *   - The final paragraph, the booking call-to-action, is kept as its own
 *     separate paragraph block so it still reads as a distinct new line.
 * Sanitized with wp_kses_post (keeps <br>, <a>, and the indent entities).
 * Returns '' for empty input.
 */
function lp_news_text_to_blocks($text)
{
	$clean = lp_news_clean_text($text);
	if ('' === $clean) {
		return '';
	}
	return lp_news_build_book_blocks(explode("\n\n", $clean));
}

/**
 * Build the book-style block markup from an ordered list of paragraph strings.
 * Shared by the publish path (lp_news_text_to_blocks) and the reformat tool
 * (which recovers paragraphs from existing post content). Returns '' when empty.
 */
function lp_news_build_book_blocks($paras)
{
	$paras = array_values(array_filter(array_map('trim', (array) $paras), function ($p) {
		return '' !== $p;
	}));
	if (empty($paras)) {
		return '';
	}

	// Last paragraph = the booking CTA; keep it as its own block when there's
	// more than one paragraph so it stays a distinct new line.
	$cta  = count($paras) >= 2 ? array_pop($paras) : '';
	$body = '';
	foreach ($paras as $i => $para) {
		$body .= (0 === $i) ? $para : '<br>' . LP_NEWS_INDENT . $para;
	}

	$blocks = array();
	if ('' !== $body) {
		$blocks[] = "<!-- wp:paragraph -->\n<p>" . wp_kses_post($body) . "</p>\n<!-- /wp:paragraph -->";
	}
	if ('' !== $cta) {
		$blocks[] = "<!-- wp:paragraph -->\n<p>" . wp_kses_post($cta) . "</p>\n<!-- /wp:paragraph -->";
	}
	return implode("\n\n", $blocks);
}

/**
 * Recover the ordered paragraph strings and the trailing "Read more" buttons
 * block from a post's existing content, so the body can be re-rendered in book
 * style. Handles both the legacy layout (one <p> block per paragraph) and the
 * new layout (one <p> with <br>-separated paragraphs + a CTA block): every <p>
 * before the buttons block is collected and split on <br>, with leading indent
 * entities/whitespace stripped. Returns array($paragraphs, $buttons_html).
 */
function lp_news_content_to_paragraphs($content)
{
	$content = (string) $content;
	$buttons = '';
	if (preg_match('/<!-- wp:buttons -->.*?<!-- \/wp:buttons -->/s', $content, $m)) {
		$buttons = $m[0];
		$content = str_replace($buttons, '', $content);
	}

	$paras = array();
	if (preg_match_all('/<p\b[^>]*>(.*?)<\/p>/is', $content, $matches)) {
		foreach ($matches[1] as $inner) {
			foreach (preg_split('/<br\s*\/?>/i', $inner) as $piece) {
				// Strip a leading first-line indent (em spaces, nbsp, or whitespace).
				$piece = preg_replace('/^(?:&emsp;|&nbsp;|&#8195;|&#160;|\s)+/i', '', $piece);
				$piece = trim($piece);
				if ('' !== $piece) {
					$paras[] = $piece;
				}
			}
		}
	}
	return array($paras, $buttons);
}

/**
 * The red "Read more" Gutenberg button block, pointed at $url. Vivid-red
 * background, white text, square corners (border-radius:0). Render-only classes
 * are deliberately omitted so Gutenberg doesn't flag the block as invalid.
 */
function lp_news_button_html($url, $label = 'Read more')
{
	$href  = esc_url($url);
	$label = '' !== trim((string) $label) ? esc_html($label) : 'Read more';
	return "\n<!-- wp:buttons -->\n"
		. '<div class="wp-block-buttons">'
		. '<!-- wp:button {"backgroundColor":"vivid-red","textColor":"white","style":{"border":{"radius":"0px"}}} -->'
		. "\n" . '<div class="wp-block-button"><a class="wp-block-button__link has-white-color has-vivid-red-background-color has-text-color has-background wp-element-button" href="' . $href . '" style="border-radius:0px">' . $label . '</a></div>' . "\n"
		. '<!-- /wp:button -->'
		. '</div>'
		. "\n<!-- /wp:buttons -->\n";
}

/**
 * Download an image from a URL into the media library and attach it to a post.
 * Returns the attachment ID or a WP_Error.
 */
function lp_news_sideload_url($url, $parent_id)
{
	require_once ABSPATH . 'wp-admin/includes/file.php';
	require_once ABSPATH . 'wp-admin/includes/media.php';
	require_once ABSPATH . 'wp-admin/includes/image.php';

	$tmp = download_url($url, 30);
	if (is_wp_error($tmp)) {
		return $tmp;
	}

	// Derive a sane filename; download_url() gives a tmp path with no extension.
	$name = basename(parse_url($url, PHP_URL_PATH));
	if ('' === $name || false === strpos($name, '.')) {
		$name = 'news-image.jpg';
	}
	$name = sanitize_file_name($name);

	$file_array = array('name' => $name, 'tmp_name' => $tmp);
	$att_id     = media_handle_sideload($file_array, $parent_id);

	if (is_wp_error($att_id)) {
		@unlink($tmp);
		return $att_id;
	}
	return $att_id;
}

/* -------------------------------------------------------------------------- *
 *  Page-settings template
 * -------------------------------------------------------------------------- */

/**
 * Post-meta keys that must NEVER be copied from the template post, WordPress
 * internals and the per-post values this plugin manages itself (dedup key,
 * featured image, edit locks, slugs, etc.). Everything else on the template, 
 * i.e. the theme's Header/Footer/Sidebar/Layout "Page settings" meta, is copied.
 */
function lp_news_template_meta_blocklist()
{
	return array(
		'_lp_news_key',
		'_thumbnail_id',
		'_wp_attached_file',
		'_wp_attachment_metadata',
		'_wp_attachment_image_alt',
		'_edit_lock',
		'_edit_last',
		'_wp_old_slug',
		'_wp_old_date',
		'_wp_desired_post_slug',
		'_pingme',
		'_encloseme',
		'_wp_trash_meta_status',
		'_wp_trash_meta_time',
	);
}

/**
 * Copy the theme's per-post "Page settings" (Header/Footer/Sidebar/Layout/Post
 * date/etc.) from the configured template post onto $post_id, so LP News posts
 * match the desired layout instead of the theme's global defaults.
 *
 * The template post's ID is stored in the LP_NEWS_OPTION_TEMPLATE option; the
 * site admin configures ONE post with the wanted settings and enters its ID on
 * the LP News settings page. Copying meta wholesale (minus the blocklist) is
 * deliberately theme-agnostic: we don't need to know the theme's private meta
 * keys or their value encodings, and it keeps working across theme updates.
 * No-op when unset, when the template is missing, or when it points at $post_id.
 */
function lp_news_copy_template_meta($post_id)
{
	$tpl = (int) get_option(LP_NEWS_OPTION_TEMPLATE, 0);
	if ($tpl <= 0 || $tpl === (int) $post_id) {
		return;
	}
	if (! get_post($tpl)) {
		return;
	}

	$blocklist = lp_news_template_meta_blocklist();
	$meta = get_post_meta($tpl); // key => array of raw (possibly serialized) values
	foreach ($meta as $key => $values) {
		if (in_array($key, $blocklist, true)) {
			continue;
		}
		delete_post_meta($post_id, $key);
		foreach ((array) $values as $value) {
			// get_post_meta() returns raw serialized strings; unserialize so
			// add_post_meta() stores arrays/objects correctly (it re-serializes).
			add_post_meta($post_id, $key, maybe_unserialize($value));
		}
	}
}

/* -------------------------------------------------------------------------- *
 *  Settings page
 * -------------------------------------------------------------------------- */

add_action('admin_menu', function () {
	add_options_page(
		'LP News',
		'LP News',
		'manage_options',
		'lp-news',
		'lp_news_settings_page'
	);
});

/**
 * Re-render the body of existing LP News posts (identified by the `_lp_news_key`
 * meta) into the current book style. Returns the number of posts updated. Only
 * touches the given statuses (drafts by default, never published posts unless
 * asked) and only rewrites when at least one paragraph was recovered.
 */
function lp_news_reformat_existing($statuses = array('draft', 'pending'))
{
	$ids = get_posts(array(
		'post_type'   => 'post',
		'post_status' => $statuses,
		'numberposts' => -1,
		'fields'      => 'ids',
		'meta_key'    => '_lp_news_key',
	));

	$updated = 0;
	foreach ($ids as $id) {
		$post = get_post($id);
		if (! $post) {
			continue;
		}
		list($paras, $buttons) = lp_news_content_to_paragraphs($post->post_content);
		if (empty($paras)) {
			continue;
		}
		$content = lp_news_build_book_blocks($paras);
		if ('' !== $buttons) {
			$content = rtrim($content) . "\n" . $buttons;
		}
		// Compare trimmed so a repeat run on an already-book-style post is a no-op
		// (avoids piling up pointless revisions on every click).
		if (trim($content) !== trim($post->post_content)) {
			wp_update_post(array('ID' => $id, 'post_content' => $content));
			$updated++;
		}
	}
	return $updated;
}

/**
 * Copy the current template post's Page settings onto a specific list of post
 * IDs (targeted fix). Returns the number updated. No-op with no template set;
 * skips the template post itself, missing posts, and non-positive IDs.
 */
function lp_news_apply_template_to_ids($ids)
{
	$tpl = (int) get_option(LP_NEWS_OPTION_TEMPLATE, 0);
	if ($tpl <= 0 || ! get_post($tpl)) {
		return 0;
	}
	$updated = 0;
	foreach ((array) $ids as $id) {
		$id = (int) $id;
		if ($id <= 0 || $id === $tpl || ! get_post($id)) {
			continue;
		}
		lp_news_copy_template_meta($id);
		$updated++;
	}
	return $updated;
}

/**
 * Copy the current template post's Page settings onto EVERY existing LP News
 * post (across all managed statuses, published included, since the goal is to
 * fix posts already on the site). Returns the number updated. No-op with no
 * template configured. The template post itself is skipped.
 */
function lp_news_apply_template_to_existing()
{
	$tpl = (int) get_option(LP_NEWS_OPTION_TEMPLATE, 0);
	if ($tpl <= 0 || ! get_post($tpl)) {
		return 0;
	}
	$ids = get_posts(array(
		'post_type'   => 'post',
		'post_status' => array('publish', 'future', 'draft', 'pending', 'private'),
		'numberposts' => -1,
		'fields'      => 'ids',
		'meta_key'    => '_lp_news_key',
	));
	$updated = 0;
	foreach ($ids as $id) {
		if ((int) $id === $tpl) {
			continue;
		}
		lp_news_copy_template_meta($id);
		$updated++;
	}
	return $updated;
}

function lp_news_settings_page()
{
	if (! current_user_can('manage_options')) {
		return;
	}

	if (isset($_POST['lpnews_apply_ids']) && check_admin_referer('lpnews_apply_ids')) {
		if ((int) get_option(LP_NEWS_OPTION_TEMPLATE, 0) <= 0) {
			echo '<div class="notice notice-warning"><p>Set a template post ID first, then apply it to specific posts.</p></div>';
		} else {
			$raw = isset($_POST['lpnews_apply_ids_field']) ? (string) $_POST['lpnews_apply_ids_field'] : '';
			$ids = array_filter(array_map('intval', preg_split('/[\s,]+/', trim($raw))));
			if (empty($ids)) {
				echo '<div class="notice notice-warning"><p>Enter one or more post IDs (comma- or space-separated).</p></div>';
			} else {
				$n = lp_news_apply_template_to_ids($ids);
				echo '<div class="notice notice-success"><p>Applied Page settings to ' . (int) $n . ' post(s).</p></div>';
			}
		}
	}

	if (isset($_POST['lpnews_apply_template']) && check_admin_referer('lpnews_apply_template')) {
		if ((int) get_option(LP_NEWS_OPTION_TEMPLATE, 0) <= 0) {
			echo '<div class="notice notice-warning"><p>Set a template post ID first, then apply it to existing posts.</p></div>';
		} else {
			$n = lp_news_apply_template_to_existing();
			echo '<div class="notice notice-success"><p>Applied Page settings to ' . (int) $n . ' existing LP News post(s).</p></div>';
		}
	}

	if (isset($_POST['lpnews_template']) && check_admin_referer('lpnews_template')) {
		$tpl = isset($_POST['lpnews_template_id']) ? max(0, (int) $_POST['lpnews_template_id']) : 0;
		update_option(LP_NEWS_OPTION_TEMPLATE, $tpl);
		echo '<div class="notice notice-success"><p>' . ($tpl > 0
			? 'Saved. New posts will copy Page settings from post #' . $tpl . '.'
			: 'Cleared. New posts will use the theme default Page settings.') . '</p></div>';
	}

	if (isset($_POST['lpnews_reformat']) && check_admin_referer('lpnews_reformat')) {
		$n = lp_news_reformat_existing(array('draft', 'pending'));
		echo '<div class="notice notice-success"><p>Reformatted ' . (int) $n . ' LP News draft post(s) into book style.</p></div>';
	}

	if (isset($_POST['lpnews_regen_secret']) && check_admin_referer('lpnews_regen_secret')) {
		if (defined('LP_NEWS_SECRET')) {
			echo '<div class="notice notice-warning"><p>Secret is pinned by the LP_NEWS_SECRET constant in wp-config.php; cannot regenerate here.</p></div>';
		} else {
			update_option(LP_NEWS_OPTION_SECRET, wp_generate_password(48, false, false));
			echo '<div class="notice notice-success"><p>Generated a new secret. Update LP_NEWS_SECRET in the Python job.</p></div>';
		}
	}

	$secret      = lp_news_get_secret();
	$publish_url = esc_url(rest_url('lp-news/v1/publish-news'));
?>
	<div class="wrap">
		<h1>LP News</h1>

		<h2>Status</h2>
		<table class="form-table">
			<tr>
				<th>Publish URL</th>
				<td><code><?php echo $publish_url; ?></code></td>
			</tr>
			<tr>
				<th>Post type</th>
				<td>Standard <code>post</code>, created as <strong>draft</strong> for review.</td>
			</tr>
		</table>

		<h2>Page settings template</h2>
		<p>New posts copy their theme "Page settings" (Header, Footer, Sidebar, Layout, Post date, etc.) from a template post you configure once. Set up a single post with exactly the layout you want, then enter its ID here. Leave blank (0) to use the theme defaults.</p>
		<?php $tpl_id = (int) get_option(LP_NEWS_OPTION_TEMPLATE, 0); ?>
		<form method="post">
			<?php wp_nonce_field('lpnews_template'); ?>
			<input type="hidden" name="lpnews_template" value="1">
			<p>
				<label>Template post ID:
					<input type="number" min="0" name="lpnews_template_id" value="<?php echo esc_attr($tpl_id); ?>" class="small-text">
				</label>
				<?php if ($tpl_id > 0) : ?>
					&nbsp;<a href="<?php echo esc_url(get_edit_post_link($tpl_id)); ?>" target="_blank">edit template post #<?php echo (int) $tpl_id; ?></a>
				<?php endif; ?>
			</p>
			<?php submit_button('Save template', 'secondary'); ?>
		</form>
		<p>Apply the saved template's Page settings to <strong>specific posts</strong>, enter one or more post IDs (comma- or space-separated). Use this for a targeted fix:</p>
		<form method="post">
			<?php wp_nonce_field('lpnews_apply_ids'); ?>
			<input type="hidden" name="lpnews_apply_ids" value="1">
			<p>
				<label>Post ID(s):
					<input type="text" name="lpnews_apply_ids_field" value="" class="regular-text" placeholder="e.g. 10999, 11024">
				</label>
			</p>
			<?php submit_button('Apply to these posts', 'secondary'); ?>
		</form>
		<p>Or apply the saved template's Page settings to <strong>all existing</strong> LP News posts (published included, this fixes every post already on the site that is missing these settings):</p>
		<form method="post" onsubmit="return confirm('Apply the template Page settings to all existing LP News posts, including published ones?');">
			<?php wp_nonce_field('lpnews_apply_template'); ?>
			<input type="hidden" name="lpnews_apply_template" value="1">
			<?php submit_button('Apply to all existing posts', 'secondary'); ?>
		</form>

		<h2>Maintenance</h2>
		<p>Re-render existing LP News <strong>draft</strong> posts into the current "like a book" layout (indented paragraphs, no inter-paragraph gaps). Published posts are not touched. Post revisions are kept, so this is reversible.</p>
		<form method="post">
			<?php wp_nonce_field('lpnews_reformat'); ?>
			<input type="hidden" name="lpnews_reformat" value="1">
			<?php submit_button('Reformat existing drafts', 'secondary'); ?>
		</form>
		<p><em>Note: this re-renders the body text only; it does not change Page settings. New posts pick up the template above automatically.</em></p>

		<h2>Secret</h2>
		<p>Send this as the <code>X-LP-News-Secret</code> header. Set it as <code>LP_NEWS_SECRET</code> in the LP Content Engine's environment.</p>
		<p><input type="text" class="regular-text" readonly value="<?php echo esc_attr($secret); ?>" onclick="this.select()"></p>
		<?php if (defined('LP_NEWS_SECRET')) : ?>
			<p><em>Pinned by the LP_NEWS_SECRET constant in wp-config.php.</em></p>
		<?php else : ?>
			<form method="post">
				<?php wp_nonce_field('lpnews_regen_secret'); ?>
				<input type="hidden" name="lpnews_regen_secret" value="1">
				<?php submit_button('Regenerate secret', 'secondary'); ?>
			</form>
		<?php endif; ?>
	</div>
<?php
}
