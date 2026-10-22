<?php

/**
 * The LP Content admin screen: one top-level menu, six tabs.
 */

if (! defined('ABSPATH')) {
	exit;
}

add_action('admin_menu', function () {
	add_menu_page(
		'LP Content',
		'LP Content',
		lp_dash_capability(),
		'lp-content',
		'lp_dash_render_page',
		'dashicons-megaphone',
		26
	);
});

function lp_dash_tabs()
{
	return array(
		'social'    => 'Social queue',
		'website'   => 'Website posts',
		'compose'   => 'Compose',
		'assistant' => 'Assistant',
		'analytics' => 'Analytics',
		'settings'  => 'Settings',
	);
}

function lp_dash_tab_url($tab)
{
	return admin_url('admin.php?page=lp-content&tab=' . rawurlencode($tab));
}

/**
 * Format an ISO timestamp in the site's timezone, or a dash.
 */
function lp_dash_when($iso)
{
	if (! $iso) {
		return '<span style="color:#888">not scheduled</span>';
	}
	$ts = strtotime($iso);
	if (! $ts) {
		return esc_html($iso);
	}
	return esc_html(wp_date('D j M Y, g:ia', $ts));
}

function lp_dash_badge($label, $color)
{
	return '<span style="display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;'
		. 'line-height:18px;background:' . $color . ';color:#fff;white-space:nowrap">'
		. esc_html($label) . '</span>';
}

function lp_dash_status_badge($status)
{
	$map = array(
		'draft'     => array('Draft', '#d63638'),
		'scheduled' => array('Queued', '#996800'),
		'sent'      => array('Sent', '#2271b1'),
		'pending'   => array('Pending', '#d63638'),
		'future'    => array('Scheduled', '#996800'),
		'publish'   => array('Published', '#2271b1'),
	);
	$m = isset($map[$status]) ? $map[$status] : array($status, '#666');
	return lp_dash_badge($m[0], $m[1]);
}

function lp_dash_render_page()
{
	if (! current_user_can(lp_dash_capability())) {
		wp_die('You do not have permission to view this page.');
	}

	$tab  = isset($_GET['tab']) ? sanitize_key($_GET['tab']) : 'social';
	$tabs = lp_dash_tabs();
	if (! isset($tabs[$tab])) {
		$tab = 'social';
	}

	// Settings writes happen before anything renders, so the page below shows
	// the state after the save rather than before it. The notice is held back
	// and echoed inside .wrap, where WordPress expects admin notices to sit.
	$notice = '';
	$notice_type = 'success';
	if ($tab === 'settings') {
		$notice = lp_dash_handle_settings_post();
	}
	$acted = lp_dash_handle_action();
	if ($acted) {
		$notice      = $acted['message'];
		$notice_type = $acted['ok'] ? 'success' : 'error';
	}
	if (isset($_GET['refresh']) && $_GET['refresh'] === '1') {
		lp_dash_buffer_flush_cache();
		delete_transient('lp_dash_fb_posts');
	}

	echo '<div class="wrap"><h1>LP Content</h1>';
	echo '<h2 class="nav-tab-wrapper">';
	foreach ($tabs as $slug => $label) {
		printf(
			'<a href="%s" class="nav-tab%s">%s</a>',
			esc_url(lp_dash_tab_url($slug)),
			$slug === $tab ? ' nav-tab-active' : '',
			esc_html($label)
		);
	}
	echo '</h2>';

	if ($notice !== '') {
		echo '<div class="notice notice-' . esc_attr($notice_type) . '"><p>' . esc_html($notice) . '</p></div>';
	}

	switch ($tab) {
		case 'website':
			lp_dash_render_website();
			break;
		case 'compose':
			lp_dash_render_compose();
			break;
		case 'assistant':
			lp_dash_render_assistant();
			break;
		case 'analytics':
			lp_dash_render_analytics();
			break;
		case 'settings':
			lp_dash_render_settings();
			break;
		default:
			lp_dash_render_social();
	}

	echo '</div>';
}

/* -------------------------------------------------------------------------- *
 *  Approval actions (human only)
 *
 *  Every one of these is a nonce-protected POST from a button on this page,
 *  gated on a capability. NONE of them is an assistant tool, and that is the
 *  whole point: Claude can tidy a draft, a person decides what goes live
 *  (client direction 2026-08-19).
 * -------------------------------------------------------------------------- */

/**
 * The capability needed to approve, publish or send.
 *
 * Separate from lp_dash_capability() so the agency can let staff *see* the
 * dashboard without letting them push things live:
 * add_filter('lp_dash_publish_capability', function () { return 'manage_options'; });
 */
function lp_dash_publish_capability()
{
	return (string) apply_filters('lp_dash_publish_capability', lp_dash_capability());
}

function lp_dash_can_publish()
{
	return current_user_can(lp_dash_publish_capability());
}

/**
 * Handle an approval POST. Returns array('ok'=>bool,'message'=>string) or null.
 */
function lp_dash_handle_action()
{
	if (empty($_POST['lp_dash_action'])) {
		return null;
	}
	check_admin_referer('lp_dash_action');

	if (! lp_dash_can_publish()) {
		return array('ok' => false, 'message' => 'You do not have permission to approve or publish.');
	}

	$action = sanitize_key($_POST['lp_dash_action']);
	$id     = isset($_POST['id']) ? sanitize_text_field(wp_unslash($_POST['id'])) : '';
	$when   = isset($_POST['when']) ? sanitize_text_field(wp_unslash($_POST['when'])) : '';

	switch ($action) {
		case 'approve':
			$res = lp_dash_buffer_approve($id, $when !== '' ? $when : null);
			$ok_msg = 'Approved. It is queued to send.';
			break;
		case 'send_now':
			$res = lp_dash_buffer_send_now($id);
			$ok_msg = 'Sent to the channel now.';
			break;
		case 'unapprove':
			$res = lp_dash_buffer_unapprove($id);
			$ok_msg = 'Pulled back to draft.';
			break;
		case 'publish_article':
			$res = lp_dash_article_publish((int) $id);
			$ok_msg = 'Article published.';
			break;
		case 'schedule_article':
			$res = lp_dash_article_schedule((int) $id, $when);
			$ok_msg = 'Article scheduled.';
			break;
		case 'unpublish_article':
			$res = lp_dash_article_unpublish((int) $id);
			$ok_msg = 'Article moved back to draft.';
			break;
		default:
			return array('ok' => false, 'message' => 'Unknown action.');
	}

	$ok = ($res === true);
	lp_dash_audit('human:' . $action, array('id' => $id, 'when' => $when), $ok, $ok ? '' : $res['error']);
	return array('ok' => $ok, 'message' => $ok ? $ok_msg : $res['error']);
}

/**
 * One button, as its own tiny POST form. Forms rather than links because these
 * change state, and a link that publishes can be fired by a prefetch or a
 * crawler following it out of the admin.
 */
function lp_dash_action_button($action, $id, $label, $confirm = '', $style = '', $extra = array())
{
	$out  = '<form method="post" style="display:inline-block;margin:0 4px 4px 0">';
	$out .= wp_nonce_field('lp_dash_action', '_wpnonce', true, false);
	$out .= '<input type="hidden" name="lp_dash_action" value="' . esc_attr($action) . '">';
	$out .= '<input type="hidden" name="id" value="' . esc_attr($id) . '">';
	foreach ($extra as $k => $v) {
		$out .= '<input type="hidden" name="' . esc_attr($k) . '" value="' . esc_attr($v) . '">';
	}
	$out .= '<button type="submit" class="button ' . esc_attr($style) . '"';
	if ($confirm !== '') {
		$out .= ' onclick="return confirm(' . esc_attr(wp_json_encode($confirm)) . ')"';
	}
	$out .= '>' . esc_html($label) . '</button></form>';
	return $out;
}

/**
 * Approve-at-a-chosen-time, for a draft whose slot has passed or is missing.
 */
function lp_dash_schedule_form($action, $id, $label, $default_ts = 0)
{
	$default = gmdate('Y-m-d\TH:i', $default_ts ? $default_ts : (time() + 86400));
	$out  = '<form method="post" style="display:inline-block;margin:0 4px 4px 0">';
	$out .= wp_nonce_field('lp_dash_action', '_wpnonce', true, false);
	$out .= '<input type="hidden" name="lp_dash_action" value="' . esc_attr($action) . '">';
	$out .= '<input type="hidden" name="id" value="' . esc_attr($id) . '">';
	$out .= '<input type="datetime-local" name="when" value="' . esc_attr($default) . '" style="width:190px">';
	$out .= '<button type="submit" class="button">' . esc_html($label) . '</button></form>';
	return $out;
}

/* -------------------------------------------------------------------------- *
 *  Social queue
 * -------------------------------------------------------------------------- */

function lp_dash_render_social()
{
	if (lp_dash_buffer_token() === '') {
		echo '<div class="notice notice-warning"><p>No Buffer token yet. Add one on the '
			. '<a href="' . esc_url(lp_dash_tab_url('settings')) . '">Settings</a> tab to see the queue.</p></div>';
		return;
	}

	$posts = lp_dash_buffer_posts();
	if (isset($posts['error'])) {
		echo '<div class="notice notice-error"><p>' . esc_html($posts['error']) . '</p></div>';
		return;
	}

	$groups = array(
		'draft'     => array('Drafts, waiting for approval', array()),
		'scheduled' => array('Queued to send', array()),
		'sent'      => array('Already sent', array()),
	);
	foreach ($posts as $p) {
		if (isset($groups[$p['status']])) {
			$groups[$p['status']][1][] = $p;
		}
	}

	printf(
		'<p style="margin:12px 0"><a class="button" href="%s">Refresh from Buffer</a> '
			. '<span style="color:#666;margin-left:8px">Cached for %d seconds so the dashboard cannot eat Buffer\'s daily API budget, which the weekly run shares.</span></p>',
		esc_url(lp_dash_tab_url('social') . '&refresh=1'),
		(int) LP_DASH_CACHE_SECONDS
	);

	foreach ($groups as $status => $group) {
		list($heading, $rows) = $group;
		echo '<h2>' . esc_html($heading) . ' <span style="color:#888;font-weight:400">(' . count($rows) . ')</span></h2>';
		if (empty($rows)) {
			echo '<p style="color:#666">Nothing here.</p>';
			continue;
		}
		$actionable = ($status !== 'sent') && lp_dash_can_publish();
		echo '<table class="widefat striped" style="margin-bottom:24px"><thead><tr>'
			. '<th style="width:110px">Channel</th><th style="width:190px">'
			. ($status === 'sent' ? 'Sent' : 'Scheduled') . '</th><th>Copy</th>'
			. ($actionable ? '<th style="width:230px">Actions</th>' : '')
			. '</tr></thead><tbody>';
		foreach ($rows as $p) {
			echo '<tr><td>' . lp_dash_badge(ucfirst($p['service']), lp_dash_channel_color($p['service'])) . '</td>';
			echo '<td>' . lp_dash_when($status === 'sent' ? $p['sentAt'] : $p['dueAt']) . '</td>';
			echo '<td><div style="white-space:pre-wrap;max-width:700px">' . esc_html($p['text']) . '</div>'
				. '<code style="color:#888;font-size:11px">' . esc_html($p['id']) . '</code></td>';
			if ($actionable) {
				echo '<td>' . lp_dash_social_actions($p) . '</td>';
			}
			echo '</tr>';
		}
		echo '</tbody></table>';
	}
}

/**
 * The buttons for one social post.
 *
 * A draft whose slot is still ahead of us approves in one click. A draft whose
 * slot has passed does NOT: it gets a date picker instead, because silently
 * moving someone's post to a day they did not choose is its own kind of wrong,
 * and because handing Buffer a stale time is exactly what published three posts
 * by accident on 2026-08-03.
 */
function lp_dash_social_actions($p)
{
	$out = '';
	$due_ts = $p['dueAt'] ? strtotime($p['dueAt']) : 0;

	if ($p['status'] === 'draft') {
		if ($due_ts && $due_ts > time()) {
			$out .= lp_dash_action_button(
				'approve',
				$p['id'],
				'Approve',
				'Approve this ' . $p['service'] . ' post? It will send at its scheduled time.',
				'button-primary'
			);
		} else {
			$out .= '<div style="color:#996800;font-size:12px;margin-bottom:4px">'
				. ($due_ts ? 'Its slot has passed.' : 'No time set.') . ' Pick one to approve:</div>';
			$out .= lp_dash_schedule_form('approve', $p['id'], 'Approve');
		}
		$out .= lp_dash_action_button(
			'send_now',
			$p['id'],
			'Send now',
			'Send this ' . $p['service'] . ' post immediately? This publishes it and cannot be undone.'
		);
	} elseif ($p['status'] === 'scheduled') {
		$out .= lp_dash_action_button(
			'unapprove',
			$p['id'],
			'Back to draft',
			'Pull this post out of the queue and back to draft?'
		);
		$out .= lp_dash_action_button(
			'send_now',
			$p['id'],
			'Send now',
			'Send this ' . $p['service'] . ' post immediately, ahead of its scheduled time? This cannot be undone.'
		);
	}
	return $out;
}

function lp_dash_channel_color($service)
{
	$map = array('linkedin' => '#0a66c2', 'instagram' => '#c13584', 'facebook' => '#1877f2');
	return isset($map[$service]) ? $map[$service] : '#666';
}

/* -------------------------------------------------------------------------- *
 *  Website posts
 * -------------------------------------------------------------------------- */

function lp_dash_render_website()
{
	$groups = array(
		'Drafts, waiting for review' => array('draft', 'pending'),
		'Scheduled to publish'       => array('future'),
		'Published'                  => array('publish'),
	);

	foreach ($groups as $heading => $statuses) {
		$rows = lp_dash_articles($statuses, 40);
		echo '<h2>' . esc_html($heading) . ' <span style="color:#888;font-weight:400">(' . count($rows) . ')</span></h2>';
		if (empty($rows)) {
			echo '<p style="color:#666">Nothing here.</p>';
			continue;
		}
		$actionable = lp_dash_can_publish();
		echo '<table class="widefat striped" style="margin-bottom:24px"><thead><tr>'
			. '<th>Title</th><th style="width:100px">Status</th><th style="width:170px">Date</th>'
			. '<th style="width:170px">Categories</th><th style="width:80px">Source</th>'
			. ($actionable ? '<th style="width:250px">Actions</th>' : '')
			. '</tr></thead><tbody>';
		foreach ($rows as $a) {
			echo '<tr><td><strong><a href="' . esc_url($a['edit_url']) . '">' . esc_html($a['title']) . '</a></strong> '
				. '<code style="color:#888;font-size:11px">#' . (int) $a['id'] . '</code></td>';
			echo '<td>' . lp_dash_status_badge($a['status']) . '</td>';
			echo '<td>' . lp_dash_when($a['date']) . '</td>';
			echo '<td>' . esc_html(implode(', ', $a['categories'])) . '</td>';
			echo '<td>' . ($a['is_lp_news'] ? 'Engine' : '<span style="color:#888">Manual</span>') . '</td>';
			if ($actionable) {
				echo '<td>' . lp_dash_article_actions($a) . '</td>';
			}
			echo '</tr>';
		}
		echo '</tbody></table>';
	}
}

/**
 * The buttons for one website article. A published one keeps only the way back,
 * which is the "we should not have sent that" button.
 */
function lp_dash_article_actions($a)
{
	$out = '';
	if ($a['status'] === 'publish') {
		return lp_dash_action_button(
			'unpublish_article',
			$a['id'],
			'Back to draft',
			'Unpublish "' . $a['title'] . '"? It disappears from the public site.'
		);
	}
	if (in_array($a['status'], array('draft', 'pending'), true)) {
		$out .= lp_dash_action_button(
			'publish_article',
			$a['id'],
			'Publish',
			'Publish "' . $a['title'] . '" to the live site now?',
			'button-primary'
		);
		$out .= '<div style="margin-top:4px">'
			. lp_dash_schedule_form('schedule_article', $a['id'], 'Schedule') . '</div>';
	} elseif ($a['status'] === 'future') {
		$out .= lp_dash_action_button(
			'publish_article',
			$a['id'],
			'Publish now',
			'Publish "' . $a['title'] . '" now, ahead of its scheduled date?',
			'button-primary'
		);
		$out .= lp_dash_action_button('unpublish_article', $a['id'], 'Back to draft');
	}
	return $out;
}

/* -------------------------------------------------------------------------- *
 *  Assistant
 * -------------------------------------------------------------------------- */

function lp_dash_render_assistant()
{
	if (lp_dash_anthropic_key() === '') {
		echo '<div class="notice notice-warning"><p>No Anthropic API key yet. Add one on the '
			. '<a href="' . esc_url(lp_dash_tab_url('settings')) . '">Settings</a> tab to use the assistant.</p></div>';
		return;
	}
	$nonce = wp_create_nonce('lp_dash_assistant');
?>
	<p style="max-width:760px">
		Ask for changes to the social queue or to website drafts in plain English. The assistant can read
		everything, and can edit, reschedule and delete items that are still <strong>drafts or still queued</strong>.
		It cannot publish, send, or change anything already live, and it cannot touch the code or these settings.
		Approving and publishing are the buttons on the <a href="<?php echo esc_url(lp_dash_tab_url('social')); ?>">Social
		queue</a> and <a href="<?php echo esc_url(lp_dash_tab_url('website')); ?>">Website posts</a> tabs, for a person to click.
	</p>
	<p style="max-width:760px;color:#666">
		For example: <em>"Shorten the Rocket Man LinkedIn draft, it's too long."</em> ·
		<em>"Move Thursday's Facebook post to Monday morning."</em> ·
		<em>"Delete the Dolly Show Instagram draft."</em>
	</p>

	<div id="lp-dash-chat" style="max-width:860px;border:1px solid #dcdcde;background:#fff;padding:14px;
		min-height:120px;max-height:460px;overflow-y:auto;margin-bottom:12px">
		<p style="color:#888;margin:0" id="lp-dash-empty">Nothing yet. Ask something below.</p>
	</div>

	<textarea id="lp-dash-input" rows="3" style="width:100%;max-width:860px"
		placeholder="What would you like changed?"></textarea>
	<p>
		<button class="button button-primary" id="lp-dash-send">Send</button>
		<button class="button" id="lp-dash-reset">Start over</button>
		<span id="lp-dash-status" style="margin-left:10px;color:#666"></span>
	</p>

	<script>
		(function() {
			var chat = document.getElementById('lp-dash-chat'),
				input = document.getElementById('lp-dash-input'),
				send = document.getElementById('lp-dash-send'),
				reset = document.getElementById('lp-dash-reset'),
				status = document.getElementById('lp-dash-status'),
				empty = document.getElementById('lp-dash-empty'),
				nonce = <?php echo wp_json_encode($nonce); ?>,
				ajax = <?php echo wp_json_encode(admin_url('admin-ajax.php')); ?>,
				busy = false;

			function bubble(who, text, tone) {
				if (empty) {
					empty.remove();
					empty = null;
				}
				var d = document.createElement('div');
				d.style.cssText = 'margin:0 0 14px;padding:10px 12px;border-radius:6px;white-space:pre-wrap;' +
					'background:' + (who === 'you' ? '#f0f0f1' : (tone === 'error' ? '#fcf0f1' : '#f6f7f7')) +
					';border-left:3px solid ' + (who === 'you' ? '#8c8f94' : (tone === 'error' ? '#d63638' : '#2271b1'));
				var b = document.createElement('strong');
				b.textContent = (who === 'you' ? 'You' : 'Assistant') + ': ';
				d.appendChild(b);
				d.appendChild(document.createTextNode(text));
				chat.appendChild(d);
				chat.scrollTop = chat.scrollHeight;
			}

			function post(action, body, done) {
				var data = new URLSearchParams(body || {});
				data.append('action', action);
				data.append('nonce', nonce);
				fetch(ajax, {
						method: 'POST',
						credentials: 'same-origin',
						headers: {
							'Content-Type': 'application/x-www-form-urlencoded'
						},
						body: data.toString()
					})
					.then(function(r) {
						return r.json();
					})
					.then(done)
					.catch(function() {
						done({
							success: false,
							data: {
								message: 'The request failed. If it was a long one it may have timed out; check the queue before retrying.'
							}
						});
					});
			}

			function submit() {
				if (busy) return;
				var msg = input.value.trim();
				if (!msg) return;
				busy = true;
				send.disabled = true;
				status.textContent = 'Thinking, this can take a few seconds...';
				bubble('you', msg);
				input.value = '';
				post('lp_dash_assistant', {
					message: msg
				}, function(res) {
					busy = false;
					send.disabled = false;
					status.textContent = '';
					if (res && res.success) {
						bubble('claude', res.data.reply);
						if (res.data.actions && res.data.actions.length) {
							status.textContent = 'Ran: ' + res.data.actions.join(', ');
						}
					} else {
						bubble('claude', (res && res.data && res.data.message) || 'Something went wrong.', 'error');
					}
				});
			}

			send.addEventListener('click', submit);
			input.addEventListener('keydown', function(e) {
				if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit();
			});
			reset.addEventListener('click', function() {
				post('lp_dash_assistant_reset', {}, function() {
					chat.innerHTML = '';
					empty = null;
					status.textContent = 'Conversation cleared.';
				});
			});
		})();
	</script>

	<h2 style="margin-top:32px">Recent changes</h2>
	<p style="color:#666">Every write the assistant makes, successful or not.</p>
	<?php lp_dash_render_audit(); ?>
<?php
}

function lp_dash_render_audit()
{
	$log = lp_dash_audit_log();
	if (empty($log)) {
		echo '<p style="color:#666">Nothing yet.</p>';
		return;
	}
	echo '<table class="widefat striped"><thead><tr><th style="width:170px">When</th>'
		. '<th style="width:120px">Who</th><th style="width:170px">Action</th>'
		. '<th style="width:70px">OK</th><th>Details</th></tr></thead><tbody>';
	foreach (array_slice($log, 0, 30) as $row) {
		echo '<tr><td>' . esc_html($row['time']) . '</td>';
		echo '<td>' . esc_html($row['user']) . '</td>';
		echo '<td><code>' . esc_html($row['tool']) . '</code></td>';
		echo '<td>' . ($row['ok'] ? '<span style="color:#00a32a">yes</span>' : '<span style="color:#d63638">no</span>') . '</td>';
		echo '<td style="font-size:12px;color:#555">' . esc_html($row['ok'] ? $row['args'] : $row['detail']) . '</td></tr>';
	}
	echo '</tbody></table>';
}

/* -------------------------------------------------------------------------- *
 *  Analytics
 * -------------------------------------------------------------------------- */

function lp_dash_render_analytics()
{
	$counts = lp_dash_article_counts();
	echo '<h2>Website posts</h2><table class="widefat striped" style="max-width:520px;margin-bottom:28px"><tbody>';
	foreach (array(
		'Drafts'             => $counts['draft'] + $counts['pending'],
		'Scheduled'          => $counts['future'],
		'Published'          => $counts['publish'],
	) as $label => $n) {
		echo '<tr><td><strong>' . esc_html($label) . '</strong></td><td>' . (int) $n . '</td></tr>';
	}
	echo '</tbody></table>';

	echo '<h2>Facebook engagement</h2>';
	if (lp_dash_meta_token() === '') {
		echo '<div class="notice notice-warning inline"><p>No Meta Page token yet. Add one on the '
			. '<a href="' . esc_url(lp_dash_tab_url('settings')) . '">Settings</a> tab.</p></div>';
	} else {
		$rows = lp_dash_facebook_posts();
		if (isset($rows['error'])) {
			echo '<div class="notice notice-error inline"><p>' . esc_html($rows['error']) . '</p></div>';
		} elseif (empty($rows)) {
			echo '<p style="color:#666">No posts returned for this Page.</p>';
		} else {
			printf(
				'<p><a class="button" href="%s">Refresh</a></p>',
				esc_url(lp_dash_tab_url('analytics') . '&refresh=1')
			);
			echo '<table class="widefat striped"><thead><tr><th>Post</th><th style="width:150px">Posted</th>'
				. '<th style="width:90px">Reactions</th><th style="width:90px">Comments</th>'
				. '<th style="width:80px">Shares</th><th style="width:80px">Total</th></tr></thead><tbody>';
			foreach (array_slice($rows, 0, 25) as $r) {
				$text = $r['message'] !== '' ? $r['message'] : '(no caption)';
				if (function_exists('mb_substr') && mb_strlen($text) > 180) {
					$text = mb_substr($text, 0, 180) . '...';
				}
				echo '<tr><td>' . ($r['url'] ? '<a href="' . esc_url($r['url']) . '" target="_blank" rel="noopener">' : '')
					. esc_html($text) . ($r['url'] ? '</a>' : '') . '</td>';
				echo '<td>' . lp_dash_when($r['created']) . '</td>';
				echo '<td>' . (int) $r['reactions'] . '</td><td>' . (int) $r['comments'] . '</td>';
				echo '<td>' . (int) $r['shares'] . '</td><td><strong>' . (int) $r['score'] . '</strong></td></tr>';
			}
			echo '</tbody></table>';
		}
	}
	?>
	<h2 style="margin-top:28px">Instagram and LinkedIn</h2>
	<div class="notice notice-info inline" style="max-width:800px">
		<p><strong>Instagram is not connected, on purpose.</strong> Pulling IG numbers requires linking the
			Facebook Page to the Instagram account, which is a UI-only operation in Meta Business settings with no
			API equivalent, and needs full control of the Page plus admin on the IG account. That was looked at on
			2026-07-31 and judged not worth the access wrangling. If IG engagement ever diverges meaningfully from
			Facebook's, the way in is Instagram API with Instagram Login, not the Business Portfolio route.</p>
		<p><strong>LinkedIn has no API access.</strong> It was applied for and not approved. Numbers come from a
			manual CSV export from the Company Page admin (Analytics, then Updates, then Export), which the weekly
			job reads. There is nothing to show here until that export is refreshed.</p>
	</div>
<?php
}

/* -------------------------------------------------------------------------- *
 *  Settings
 * -------------------------------------------------------------------------- */

/**
 * Apply a settings save. Returns a message for the caller to render, or ''.
 */
function lp_dash_handle_settings_post()
{
	if (! isset($_POST['lp_dash_settings'])) {
		return '';
	}
	if (! current_user_can(lp_dash_capability()) || ! check_admin_referer('lp_dash_settings')) {
		return '';
	}

	$fields = array(
		LP_DASH_OPT_BUFFER_TOKEN  => 'LP_DASH_BUFFER_TOKEN',
		LP_DASH_OPT_ANTHROPIC_KEY => 'LP_DASH_ANTHROPIC_KEY',
		LP_DASH_OPT_META_TOKEN    => 'LP_DASH_META_TOKEN',
	);
	foreach ($fields as $option => $constant) {
		if (lp_dash_is_pinned($constant)) {
			continue; // A constant wins; silently writing an option would mislead.
		}
		if (! isset($_POST[$option])) {
			continue;
		}
		$value = trim((string) wp_unslash($_POST[$option]));
		// An untouched masked field must not overwrite the stored secret.
		if ($value === '' || strpos($value, '****') !== false) {
			continue;
		}
		update_option($option, $value, false);
	}

	if (isset($_POST['lp_dash_clear'])) {
		foreach (array_keys($fields) as $option) {
			if (! lp_dash_is_pinned($fields[$option])) {
				delete_option($option);
			}
		}
	}

	if (isset($_POST[LP_DASH_OPT_MODEL])) {
		update_option(LP_DASH_OPT_MODEL, sanitize_text_field(wp_unslash($_POST[LP_DASH_OPT_MODEL])), false);
	}

	lp_dash_buffer_flush_cache();
	delete_transient('lp_dash_buffer_org');
	delete_transient('lp_dash_buffer_channels');
	delete_transient('lp_dash_fb_posts');
	return isset($_POST['lp_dash_clear']) ? 'Stored credentials cleared.' : 'Saved.';
}

function lp_dash_render_settings()
{
	$rows = array(
		array(LP_DASH_OPT_BUFFER_TOKEN,  'LP_DASH_BUFFER_TOKEN',  'Buffer API token', 'Reads the social queue and applies the assistant\'s edits.'),
		array(LP_DASH_OPT_ANTHROPIC_KEY, 'LP_DASH_ANTHROPIC_KEY', 'Anthropic API key', 'Powers the Assistant tab. Without it the dashboard is read-only.'),
		array(LP_DASH_OPT_META_TOKEN,    'LP_DASH_META_TOKEN',    'Meta Page access token', 'Facebook engagement on the Analytics tab. Optional.'),
	);
?>
	<form method="post">
		<?php wp_nonce_field('lp_dash_settings'); ?>
		<input type="hidden" name="lp_dash_settings" value="1">
		<table class="form-table">
			<?php foreach ($rows as $row) :
				list($option, $constant, $label, $help) = $row;
				$pinned = lp_dash_is_pinned($constant);
				$value  = lp_dash_secret($option, $constant);
			?>
				<tr>
					<th scope="row"><label for="<?php echo esc_attr($option); ?>"><?php echo esc_html($label); ?></label></th>
					<td>
						<?php if ($pinned) : ?>
							<code><?php echo esc_html(lp_dash_mask($value)); ?></code>
							<p class="description">Pinned by the <code><?php echo esc_html($constant); ?></code> constant in wp-config.php. Edit it there.</p>
						<?php else : ?>
							<input type="text" class="regular-text" id="<?php echo esc_attr($option); ?>"
								name="<?php echo esc_attr($option); ?>" autocomplete="off"
								placeholder="<?php echo $value !== '' ? esc_attr(lp_dash_mask($value)) : 'not set'; ?>">
							<p class="description">
								<?php echo esc_html($help); ?>
								<?php if ($value !== '') : ?>
									Leave blank to keep the current value.
								<?php endif; ?>
							</p>
						<?php endif; ?>
					</td>
				</tr>
			<?php endforeach; ?>
			<tr>
				<th scope="row"><label for="<?php echo esc_attr(LP_DASH_OPT_MODEL); ?>">Assistant model</label></th>
				<td>
					<input type="text" class="regular-text" id="<?php echo esc_attr(LP_DASH_OPT_MODEL); ?>"
						name="<?php echo esc_attr(LP_DASH_OPT_MODEL); ?>"
						value="<?php echo esc_attr(lp_dash_model()); ?>">
					<p class="description">Default <code><?php echo esc_html(LP_DASH_DEFAULT_MODEL); ?></code>.</p>
				</td>
			</tr>
		</table>
		<p class="submit">
			<button type="submit" class="button button-primary">Save</button>
			<button type="submit" name="lp_dash_clear" value="1" class="button"
				onclick="return confirm('Clear the stored Buffer, Anthropic and Meta credentials?')">Clear stored credentials</button>
		</p>
	</form>

	<h2>Where these are kept</h2>
	<p style="max-width:800px">
		These are stored as WordPress options, so anyone with admin access to this site can change them and they
		sit in the database and in database backups. To move one out of the database, define a constant of the same
		name in <code>wp-config.php</code>: it takes precedence over whatever is stored here, and this page will
		show it as pinned rather than editable.
	</p>
	<p style="max-width:800px;color:#666">
		The assistant is given a fixed set of tools and none of them can read a credential, publish, send, or change
		a published item. What it can do is edit, reschedule and delete drafts and queued items, and every one of
		those writes is logged on the Assistant tab.
	</p>
<?php
}
