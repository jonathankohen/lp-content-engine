<?php

/**
 * The assistant: an Anthropic tool-use loop scoped to the queue and drafts.
 *
 * WHAT KEEPS THIS SAFE IS THE TOOL LIST, NOT THE PROMPT. Claude can only do
 * what a tool lets it do, and there is no tool here that publishes, sends,
 * touches code, changes settings, or reads a credential. Adding one is the only
 * way to widen its reach, so think hard before you do.
 *
 * Staff CAN approve, send and publish, through the buttons on the Social queue
 * and Website posts tabs (lp_dash_handle_action in lp-dash-page.php). Those are
 * nonce-protected human actions and they are deliberately NOT mirrored here.
 * The split is the client's: Claude tidies drafts, a person decides what goes
 * live.
 *
 * Raw HTTP via wp_remote_post rather than the Anthropic PHP SDK: this plugin is
 * deployed by uploading a folder to WordPress, and there is no Composer step in
 * that pipeline to install a vendor tree.
 */

if (! defined('ABSPATH')) {
	exit;
}

define('LP_DASH_ANTHROPIC_URL', 'https://api.anthropic.com/v1/messages');
define('LP_DASH_ANTHROPIC_VERSION', '2023-06-01');

/**
 * The tools. Every one is read-only or writes only to a draft or queued item,
 * and each write re-checks the target's status in PHP before touching it.
 */
function lp_dash_tools()
{
	return array(
		array(
			'name'         => 'list_social_posts',
			'description'  => 'List Buffer social posts for LinkedIn, Instagram and Facebook. Use this before editing anything, to get post ids and see current copy.',
			'input_schema' => array(
				'type'       => 'object',
				'properties' => array(
					'status'  => array(
						'type'        => 'string',
						'enum'        => array('draft', 'scheduled', 'sent', 'all'),
						'description' => 'draft = awaiting approval, scheduled = queued to send, sent = already published. Default all.',
					),
					'service' => array(
						'type'        => 'string',
						'enum'        => array('linkedin', 'instagram', 'facebook', 'all'),
						'description' => 'Channel filter. Default all.',
					),
				),
				'required'   => array(),
			),
		),
		array(
			'name'         => 'list_articles',
			'description'  => 'List website news articles on loveproductions.com. draft/pending are awaiting review, future is scheduled to publish, publish is already live.',
			'input_schema' => array(
				'type'       => 'object',
				'properties' => array(
					'status' => array(
						'type'        => 'string',
						'enum'        => array('draft', 'pending', 'future', 'publish', 'all'),
						'description' => 'Default all.',
					),
				),
				'required'   => array(),
			),
		),
		array(
			'name'         => 'read_article',
			'description'  => 'Read one website article in full: its title, status, categories and body text. Do this before rewriting a body so you edit the real copy rather than guessing at it.',
			'input_schema' => array(
				'type'       => 'object',
				'properties' => array(
					'id' => array('type' => 'integer', 'description' => 'WordPress post id.'),
				),
				'required'   => array('id'),
			),
		),
		array(
			'name'         => 'update_social_post',
			'description'  => 'Change the copy and/or the scheduled time of a Buffer draft or queued post. Cannot touch a post that has already been sent, and never publishes: a draft stays a draft, a queued post stays queued.',
			'input_schema' => array(
				'type'       => 'object',
				'properties' => array(
					'id'           => array('type' => 'string', 'description' => 'Buffer post id from list_social_posts.'),
					'text'         => array('type' => 'string', 'description' => 'The full replacement copy. Omit to leave the copy alone.'),
					'scheduled_at' => array('type' => 'string', 'description' => 'New time as ISO 8601 UTC, e.g. 2026-08-25T14:00:00Z. Omit to keep the existing time. A queued post requires a future time.'),
				),
				'required'   => array('id'),
			),
		),
		array(
			'name'         => 'delete_social_post',
			'description'  => 'Delete a Buffer draft or queued post. This is permanent and Buffer has no undo, so only use it when the person has clearly asked for that specific post to be removed.',
			'input_schema' => array(
				'type'       => 'object',
				'properties' => array(
					'id' => array('type' => 'string', 'description' => 'Buffer post id from list_social_posts.'),
				),
				'required'   => array('id'),
			),
		),
		array(
			'name'         => 'update_article',
			'description'  => 'Change the title, body or categories of a website article that is still a draft or scheduled. Cannot edit a published article and cannot publish anything. The body is plain text with a blank line between paragraphs; the page layout and the Read more button are preserved automatically.',
			'input_schema' => array(
				'type'       => 'object',
				'properties' => array(
					'id'         => array('type' => 'integer', 'description' => 'WordPress post id.'),
					'title'      => array('type' => 'string'),
					'body'       => array('type' => 'string', 'description' => 'Full replacement body, paragraphs separated by a blank line.'),
					'categories' => array(
						'type'        => 'array',
						'items'       => array('type' => 'string'),
						'description' => 'Category names. Only names on the plugin allowlist are applied; anything else is dropped.',
					),
				),
				'required'   => array('id'),
			),
		),
		array(
			'name'         => 'trash_article',
			'description'  => 'Move a draft or scheduled website article to the WordPress trash. Recoverable from the Trash view.',
			'input_schema' => array(
				'type'       => 'object',
				'properties' => array(
					'id' => array('type' => 'integer', 'description' => 'WordPress post id.'),
				),
				'required'   => array('id'),
			),
		),
	);
}

/**
 * Run one tool. Returns a string, which is what goes back as the tool_result.
 *
 * Every branch re-reads the target and re-checks its status rather than
 * trusting anything the model said, and every write is written to the audit log
 * whether it succeeded or not.
 */
function lp_dash_run_tool($name, $input)
{
	if (! current_user_can(lp_dash_capability())) {
		return 'Error: you do not have permission to do that.';
	}
	$input = is_array($input) ? $input : array();

	switch ($name) {

		case 'list_social_posts':
			$status  = isset($input['status']) ? $input['status'] : 'all';
			$service = isset($input['service']) ? $input['service'] : 'all';
			$posts   = lp_dash_buffer_posts();
			if (isset($posts['error'])) {
				return 'Error: ' . $posts['error'];
			}
			$rows = array();
			foreach ($posts as $p) {
				if ($status !== 'all' && $p['status'] !== $status) {
					continue;
				}
				if ($service !== 'all' && $p['service'] !== $service) {
					continue;
				}
				$rows[] = array(
					'id'      => $p['id'],
					'channel' => $p['service'],
					'status'  => $p['status'],
					'due'     => $p['dueAt'],
					'sent'    => $p['sentAt'],
					'text'    => $p['text'],
				);
			}
			if (empty($rows)) {
				return 'No Buffer posts match that filter.';
			}
			return wp_json_encode($rows);

		case 'list_articles':
			$status   = isset($input['status']) ? $input['status'] : 'all';
			$statuses = $status === 'all'
				? array('draft', 'pending', 'future', 'publish')
				: array($status);
			$rows = array();
			foreach (lp_dash_articles($statuses) as $a) {
				$rows[] = array(
					'id'         => $a['id'],
					'title'      => $a['title'],
					'status'     => $a['status'],
					'date'       => $a['date'],
					'categories' => $a['categories'],
				);
			}
			if (empty($rows)) {
				return 'No website articles match that filter.';
			}
			return wp_json_encode($rows);

		case 'read_article':
			$id   = isset($input['id']) ? (int) $input['id'] : 0;
			$post = get_post($id);
			if (! $post || $post->post_type !== 'post') {
				return 'Error: no post with id ' . $id . '.';
			}
			$cats = wp_get_post_terms($id, 'category', array('fields' => 'names'));
			return wp_json_encode(array(
				'id'         => $id,
				'title'      => get_the_title($post),
				'status'     => $post->post_status,
				'categories' => is_wp_error($cats) ? array() : $cats,
				'body'       => lp_dash_article_body_text($id),
			));

		case 'update_social_post':
			$id   = isset($input['id']) ? (string) $input['id'] : '';
			$text = array_key_exists('text', $input) ? (string) $input['text'] : null;
			$due  = array_key_exists('scheduled_at', $input) ? (string) $input['scheduled_at'] : null;
			if ($id === '') {
				return 'Error: id is required.';
			}
			$res = lp_dash_buffer_edit($id, $text, $due);
			$ok  = ($res === true);
			lp_dash_audit('update_social_post', $input, $ok, $ok ? '' : $res['error']);
			return $ok ? 'Updated Buffer post ' . $id . '.' : 'Error: ' . $res['error'];

		case 'delete_social_post':
			$id = isset($input['id']) ? (string) $input['id'] : '';
			if ($id === '') {
				return 'Error: id is required.';
			}
			$res = lp_dash_buffer_delete($id);
			$ok  = ($res === true);
			lp_dash_audit('delete_social_post', $input, $ok, $ok ? '' : $res['error']);
			return $ok ? 'Deleted Buffer post ' . $id . '.' : 'Error: ' . $res['error'];

		case 'update_article':
			$id    = isset($input['id']) ? (int) $input['id'] : 0;
			$title = array_key_exists('title', $input) ? (string) $input['title'] : null;
			$body  = array_key_exists('body', $input) ? (string) $input['body'] : null;
			$cats  = isset($input['categories']) && is_array($input['categories']) ? $input['categories'] : null;
			$res   = lp_dash_article_update($id, $title, $body, $cats);
			$ok    = ($res === true);
			lp_dash_audit('update_article', $input, $ok, $ok ? '' : $res['error']);
			return $ok ? 'Updated article ' . $id . '.' : 'Error: ' . $res['error'];

		case 'trash_article':
			$id  = isset($input['id']) ? (int) $input['id'] : 0;
			$res = lp_dash_article_trash($id);
			$ok  = ($res === true);
			lp_dash_audit('trash_article', $input, $ok, $ok ? '' : $res['error']);
			return $ok ? 'Moved article ' . $id . ' to the trash.' : 'Error: ' . $res['error'];
	}

	return 'Error: unknown tool "' . $name . '".';
}

/**
 * The system prompt. Deliberately short on style rules: the house voice lives in
 * the content skill graph in the Python repo, and duplicating a stale copy of it
 * here would be worse than saying nothing. What it does carry is the handful of
 * hard rules that a wrong answer makes publicly embarrassing.
 */
function lp_dash_system_prompt()
{
	return implode("\n", array(
		'You are the content assistant inside the Love Productions WordPress admin.',
		'Love Productions is a New York booking agency (founded 1985) representing live entertainment acts.',
		'',
		'You are helping agency staff manage social posts queued in Buffer and news articles on loveproductions.com.',
		'',
		'WHAT YOU CAN DO',
		'You can read everything, and you can edit, reschedule and delete items that are still drafts or still queued.',
		'You cannot publish or send anything, and you cannot change any published item. A human does that.',
		'If someone asks you to publish, send, approve or push something live, say plainly that you cannot, and point them at the Approve, Send now or Publish buttons on the Social queue and Website posts tabs. Those are for a person to click, by design.',
		'',
		'HOW TO WORK',
		'Always list or read the real item before you change it. Never edit from memory or from what the person recalled.',
		'When you change copy, return the FULL replacement text, not a fragment.',
		'Make exactly the change that was asked for. Do not tidy other posts you noticed along the way.',
		'If a request is ambiguous about which post it means, ask rather than guess.',
		'After you act, say briefly what you changed and which ids you touched.',
		'',
		'HOUSE RULES FOR ANY COPY YOU WRITE',
		'No em dashes or en dashes, ever. Use a comma, a colon, or a full stop and a new sentence.',
		'Never call the act "Elvis" anything. That act is "The Concert of Kings". Elvis Presley as the original artist is fine.',
		'Not every act is a tribute act. Priscilla Presley, The Platters, Tony Danza, Reza, Michael Griffin, Vitaly and Calpulli Mex Dance Co. are not tributes. Do not call a real performer a tribute act.',
		'Do not explain the business to talent buyers, and cut sentences that state the obvious. Show, do not tell.',
		'Instagram captions carry no links or email addresses, they use the fixed line "Link in bio to set up an appointment for booking." and at most 5 hashtags.',
		'LinkedIn and Facebook end with "Booking inquiries: info@loveproductions.com".',
	));
}

/**
 * One call to the Messages API. Returns the decoded body or array('error').
 */
function lp_dash_anthropic_call($messages)
{
	$key = lp_dash_anthropic_key();
	if ($key === '') {
		return array('error' => 'No Anthropic API key configured. Add one on the Settings tab.');
	}

	$body = array(
		'model'      => lp_dash_model(),
		'max_tokens' => LP_DASH_MAX_TOKENS,
		'system'     => lp_dash_system_prompt(),
		'tools'      => lp_dash_tools(),
		'messages'   => $messages,
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
	return $decoded;
}

/**
 * Drive the tool loop until Claude stops asking for tools.
 *
 * $history is the running conversation (an array of Messages API message
 * arrays). Returns array('reply' => string, 'history' => array, 'actions' =>
 * array of tool names run), or array('error' => string).
 */
function lp_dash_assistant_run($history, $user_message)
{
	$history[] = array('role' => 'user', 'content' => (string) $user_message);

	$actions = array();
	$reply   = '';

	for ($turn = 0; $turn < LP_DASH_MAX_TURNS; $turn++) {
		$response = lp_dash_anthropic_call($history);
		if (isset($response['error'])) {
			return array('error' => $response['error']);
		}

		// A safety decline arrives as HTTP 200 with stop_reason "refusal", so
		// it has to be checked before the content is read.
		if (isset($response['stop_reason']) && $response['stop_reason'] === 'refusal') {
			return array('error' => 'Claude declined to answer that one.');
		}

		$content = isset($response['content']) && is_array($response['content'])
			? $response['content'] : array();

		// The assistant turn goes back verbatim, tool_use blocks included.
		$history[] = array('role' => 'assistant', 'content' => $content);

		$tool_results = array();
		foreach ($content as $block) {
			if (! isset($block['type'])) {
				continue;
			}
			if ($block['type'] === 'text') {
				$reply .= (isset($block['text']) ? $block['text'] : '');
			} elseif ($block['type'] === 'tool_use') {
				$tool_name = isset($block['name']) ? $block['name'] : '';
				$actions[] = $tool_name;
				$result    = lp_dash_run_tool($tool_name, isset($block['input']) ? $block['input'] : array());
				$tool_results[] = array(
					'type'        => 'tool_result',
					'tool_use_id' => isset($block['id']) ? $block['id'] : '',
					'content'     => (string) $result,
				);
			}
		}

		if (empty($tool_results)) {
			return array('reply' => trim($reply), 'history' => $history, 'actions' => $actions);
		}

		// All tool results go back in ONE user message. Splitting them trains
		// the model out of making parallel calls.
		$history[] = array('role' => 'user', 'content' => $tool_results);
		$reply     = '';
	}

	return array(
		'error' => 'That took more than ' . LP_DASH_MAX_TURNS . ' steps, so I stopped. Any changes made before that point were applied. Try asking for one thing at a time.',
	);
}

/* -------------------------------------------------------------------------- *
 *  AJAX endpoint
 * -------------------------------------------------------------------------- */

add_action('wp_ajax_lp_dash_assistant', 'lp_dash_ajax_assistant');

function lp_dash_ajax_assistant()
{
	if (! current_user_can(lp_dash_capability())) {
		wp_send_json_error(array('message' => 'You do not have permission to use this.'), 403);
	}
	check_ajax_referer('lp_dash_assistant', 'nonce');

	$message = isset($_POST['message']) ? sanitize_textarea_field(wp_unslash($_POST['message'])) : '';
	if (trim($message) === '') {
		wp_send_json_error(array('message' => 'Type something first.'), 400);
	}
	if (strlen($message) > 4000) {
		wp_send_json_error(array('message' => 'That message is too long. Keep it under 4000 characters.'), 400);
	}

	// Conversation state lives in a per-user transient so follow-ups work
	// ("now do the same to the Facebook one") without a database table.
	$key     = 'lp_dash_convo_' . get_current_user_id();
	$history = get_transient($key);
	if (! is_array($history)) {
		$history = array();
	}
	if (isset($_POST['reset']) && $_POST['reset'] === '1') {
		$history = array();
	}

	$result = lp_dash_assistant_run($history, $message);

	if (isset($result['error'])) {
		wp_send_json_error(array('message' => $result['error']), 200);
	}

	// Keep the tail only. The whole point of the cap is that an admin page
	// request cannot be allowed to grow without bound.
	$trimmed = array_slice($result['history'], -24);
	set_transient($key, $trimmed, HOUR_IN_SECONDS);

	wp_send_json_success(array(
		'reply'   => $result['reply'] !== '' ? $result['reply'] : 'Done.',
		'actions' => array_values(array_unique($result['actions'])),
	));
}

add_action('wp_ajax_lp_dash_assistant_reset', 'lp_dash_ajax_assistant_reset');

function lp_dash_ajax_assistant_reset()
{
	if (! current_user_can(lp_dash_capability())) {
		wp_send_json_error(array('message' => 'Not allowed.'), 403);
	}
	check_ajax_referer('lp_dash_assistant', 'nonce');
	delete_transient('lp_dash_convo_' . get_current_user_id());
	wp_send_json_success(array('message' => 'Conversation cleared.'));
}
