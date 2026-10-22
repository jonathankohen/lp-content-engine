<?php

/**
 * Buffer GraphQL client for the LP Content dashboard.
 *
 * A faithful port of lp/buffer.py from the lp-content-engine repo. Every quirk
 * below was learned the hard way there and each one fails silently or
 * destructively if dropped, so read the comments before changing a mutation.
 */

if (! defined('ABSPATH')) {
	exit;
}

define('LP_DASH_BUFFER_URL', 'https://api.buffer.com');

/**
 * POST a GraphQL document to Buffer.
 *
 * Returns the decoded body, or array('error' => '...') so every caller has one
 * shape to check. Deliberately does NOT retry on 429: this runs inside an admin
 * page request, and sleeping 61 seconds would just hand the user a PHP timeout
 * instead of an error message.
 */
function lp_dash_buffer_gql($query, $variables = null)
{
	$token = lp_dash_buffer_token();
	if ($token === '') {
		return array('error' => 'No Buffer token configured. Add one on the Settings tab.');
	}

	$payload = array('query' => $query);
	if ($variables !== null) {
		$payload['variables'] = $variables;
	}

	$resp = wp_remote_post(LP_DASH_BUFFER_URL, array(
		'timeout' => 20,
		'headers' => array(
			'Content-Type'  => 'application/json',
			'Authorization' => 'Bearer ' . $token,
		),
		'body'    => wp_json_encode($payload),
	));

	if (is_wp_error($resp)) {
		return array('error' => 'Buffer request failed: ' . $resp->get_error_message());
	}

	$code = (int) wp_remote_retrieve_response_code($resp);
	$body = json_decode(wp_remote_retrieve_body($resp), true);

	if ($code === 429) {
		// Buffer has two distinct limits. The 24h one cannot be waited out
		// inside a page load, and it is shared with the weekly run, so say
		// which one was hit rather than offering a pointless retry.
		$window = '';
		if (is_array($body) && ! empty($body['errors'][0]['extensions']['window'])) {
			$window = (string) $body['errors'][0]['extensions']['window'];
		}
		if ($window === '24h') {
			return array('error' => 'Buffer daily API limit reached. It resets tomorrow, and the weekly run shares this budget.');
		}
		return array('error' => 'Buffer rate limited (60 requests/minute). Wait a minute and refresh.');
	}

	if ($code < 200 || $code >= 300) {
		return array('error' => 'Buffer HTTP ' . $code . '.');
	}

	if (! is_array($body)) {
		return array('error' => 'Buffer returned an unreadable response.');
	}

	if (! empty($body['errors'])) {
		$msg = isset($body['errors'][0]['message']) ? $body['errors'][0]['message'] : 'unknown GraphQL error';
		return array('error' => 'Buffer GraphQL error: ' . $msg);
	}

	return $body;
}

/**
 * The organization id, cached for 12 hours. Buffer's PostsInput wants
 * organizationId, not channelId (the latter is silently accepted and returns
 * nothing useful).
 */
function lp_dash_buffer_org_id($force = false)
{
	$key = 'lp_dash_buffer_org';
	if (! $force) {
		$cached = get_transient($key);
		if ($cached) {
			return (string) $cached;
		}
	}
	$data = lp_dash_buffer_gql('query { account { organizations { id name } } }');
	if (isset($data['error'])) {
		return '';
	}
	$orgs = isset($data['data']['account']['organizations']) ? $data['data']['account']['organizations'] : array();
	if (empty($orgs[0]['id'])) {
		return '';
	}
	set_transient($key, $orgs[0]['id'], 12 * HOUR_IN_SECONDS);
	return (string) $orgs[0]['id'];
}

/**
 * {service => channel id} for the three channels we post to, cached for 12
 * hours. A port of discover_buffer_profiles() in lp/buffer.py.
 *
 * ChannelsInput wants organizationId, same as PostsInput. First channel per
 * service wins: the agency has one of each, and picking arbitrarily among
 * duplicates would be worse than picking the first deterministically.
 */
function lp_dash_buffer_channels($force = false)
{
	$key = 'lp_dash_buffer_channels';
	if (! $force) {
		$cached = get_transient($key);
		if (is_array($cached)) {
			return $cached;
		}
	}

	$org = lp_dash_buffer_org_id($force);
	if ($org === '') {
		return array('error' => 'Could not read your Buffer organization. Check the token.');
	}

	$data = lp_dash_buffer_gql(
		'query GetChannels($input: ChannelsInput!) {
		   channels(input: $input) { id service displayName }
		 }',
		array('input' => array('organizationId' => $org))
	);
	if (isset($data['error'])) {
		return $data;
	}

	$out = array();
	foreach ((isset($data['data']['channels']) ? $data['data']['channels'] : array()) as $c) {
		$service = isset($c['service']) ? strtolower((string) $c['service']) : '';
		if (in_array($service, array('linkedin', 'instagram', 'facebook'), true) && ! isset($out[$service])) {
			$out[$service] = array(
				'id'   => (string) $c['id'],
				'name' => isset($c['displayName']) ? (string) $c['displayName'] : ucfirst($service),
			);
		}
	}
	if (empty($out)) {
		return array('error' => 'Buffer returned no LinkedIn, Instagram or Facebook channels for this organization.');
	}

	set_transient($key, $out, 12 * HOUR_IN_SECONDS);
	return $out;
}

/**
 * Create one Buffer DRAFT. Returns the new post id, or array('error').
 *
 * A port of post_draft_to_buffer() in lp/buffer.py, and the per-network rules
 * there are load-bearing rather than decorative:
 *
 *  - saveToDraft is always true. This function cannot queue or send, by design:
 *    approving is a button a person clicks on the Social queue tab.
 *  - PostTypeFacebook is post|reel|story. There is no "video" member and
 *    sending one is a hard GraphQL error, so a clip is a normal post that
 *    happens to carry a video asset.
 *  - Instagram has no native video type but reel, and shouldShareToFeed puts it
 *    in the grid as well.
 *  - AssetInput is a LIST. The single-dict form worked only because GraphQL
 *    coerces a lone value into a one-element list.
 *
 * $media is array('images' => array(url,...), 'video' => url|'',
 * 'thumbnail' => url|''). Buffer FETCHES these URLs itself, so they have to be
 * reachable from the public internet, not just from this server.
 */
function lp_dash_buffer_create($service, $text, $media = array(), $due_at_iso = null)
{
	$service = strtolower((string) $service);
	$channels = lp_dash_buffer_channels();
	if (isset($channels['error'])) {
		return $channels;
	}
	if (! isset($channels[$service])) {
		return array('error' => 'No ' . $service . ' channel connected in Buffer.');
	}

	$text = trim((string) $text);
	if ($text === '') {
		return array('error' => 'The ' . $service . ' copy is empty.');
	}

	$images    = isset($media['images']) && is_array($media['images']) ? array_values(array_filter($media['images'])) : array();
	$video     = isset($media['video']) ? trim((string) $media['video']) : '';
	$thumbnail = isset($media['thumbnail']) ? trim((string) $media['thumbnail']) : '';

	// Instagram will not accept a post with nothing to show. Buffer rejects it
	// at create time with a message staff cannot act on, so catch it here.
	if ($service === 'instagram' && $video === '' && empty($images)) {
		return array('error' => 'Instagram needs an image or a video. Attach one, or leave Instagram unticked.');
	}

	$input = array(
		'text'           => $text,
		'channelId'      => $channels[$service]['id'],
		'schedulingType' => 'automatic',
		'saveToDraft'    => true,
	);
	if ($due_at_iso) {
		$ts = strtotime($due_at_iso);
		if (! $ts) {
			return array('error' => 'Could not read the time "' . $due_at_iso . '".');
		}
		$input['mode']  = 'customScheduled';
		$input['dueAt'] = gmdate('c', $ts);
	} else {
		$input['mode'] = 'addToQueue';
	}

	$assets = array();
	if ($video !== '') {
		$va = array('url' => $video);
		if ($thumbnail !== '') {
			$va['thumbnailUrl'] = $thumbnail;
		}
		$assets = array(array('video' => $va));
	} elseif (! empty($images)) {
		foreach ($images as $u) {
			$assets[] = array('image' => array('url' => (string) $u));
		}
	}

	if ($service === 'facebook') {
		$input['metadata'] = array('facebook' => array('type' => 'post'));
	} elseif ($service === 'instagram') {
		$input['metadata'] = array('instagram' => array(
			'type'              => $video !== '' ? 'reel' : 'post',
			'shouldShareToFeed' => true,
		));
	}
	if (! empty($assets)) {
		$input['assets'] = $assets;
	}

	$data = lp_dash_buffer_gql(
		'mutation CreateDraft($input: CreatePostInput!) {
		   createPost(input: $input) {
		     ... on PostActionSuccess { post { id } }
		     ... on MutationError { message }
		   }
		 }',
		array('input' => $input)
	);
	if (isset($data['error'])) {
		return $data;
	}

	$result = isset($data['data']['createPost']) ? $data['data']['createPost'] : array();
	if (! is_array($result)) {
		$result = array();
	}
	if (isset($result['message'])) {
		return array('error' => 'Buffer refused the ' . $service . ' draft: ' . $result['message']);
	}
	if (empty($result['post']['id'])) {
		// Same rule as the edit path: an empty result means the GraphQL call
		// itself failed, and reporting a success we cannot see is worse than
		// reporting a failure that turns out to have written something.
		return array('error' => 'Buffer returned nothing for the ' . $service . ' draft, treating it as failed.');
	}

	lp_dash_buffer_flush_cache();
	return (string) $result['post']['id'];
}

/**
 * Every post Buffer will give us, normalised and cached.
 *
 * One page of 100. Buffer removed cursor pagination and the `status` filter
 * from PostsInput in a 2026 API update, so filtering happens here rather than
 * server-side, exactly as fetch_buffer_posts() does in Python.
 */
function lp_dash_buffer_posts($force = false)
{
	$key = 'lp_dash_buffer_posts';
	if (! $force) {
		$cached = get_transient($key);
		if (is_array($cached)) {
			return $cached;
		}
	}

	$org = lp_dash_buffer_org_id($force);
	if ($org === '') {
		return array('error' => 'Could not read your Buffer organization. Check the token.');
	}

	$data = lp_dash_buffer_gql(
		'query GetPosts($input: PostsInput!) {
		   posts(input: $input, first: 100) {
		     edges { node { id text status dueAt sentAt channel { service } } }
		   }
		 }',
		array('input' => array('organizationId' => $org))
	);
	if (isset($data['error'])) {
		return $data;
	}

	$edges = isset($data['data']['posts']['edges']) ? $data['data']['posts']['edges'] : array();
	$out   = array();
	foreach ($edges as $edge) {
		$node = isset($edge['node']) ? $edge['node'] : array();
		if (empty($node['id'])) {
			continue;
		}
		$out[] = array(
			'id'      => (string) $node['id'],
			'text'    => isset($node['text']) ? (string) $node['text'] : '',
			'service' => isset($node['channel']['service']) ? (string) $node['channel']['service'] : '',
			'status'  => isset($node['status']) ? (string) $node['status'] : '',
			'dueAt'   => isset($node['dueAt']) ? $node['dueAt'] : null,
			'sentAt'  => isset($node['sentAt']) ? $node['sentAt'] : null,
		);
	}

	// Soonest first, undated last: the queue reads as a schedule.
	usort($out, function ($a, $b) {
		$x = $a['dueAt'] ? $a['dueAt'] : ($a['sentAt'] ? $a['sentAt'] : '9999');
		$y = $b['dueAt'] ? $b['dueAt'] : ($b['sentAt'] ? $b['sentAt'] : '9999');
		return strcmp($x, $y);
	});

	set_transient($key, $out, LP_DASH_CACHE_SECONDS);
	return $out;
}

function lp_dash_buffer_flush_cache()
{
	delete_transient('lp_dash_buffer_posts');
}

/**
 * One post by id, from the cached set.
 */
function lp_dash_buffer_find($post_id)
{
	$posts = lp_dash_buffer_posts();
	if (isset($posts['error'])) {
		return $posts;
	}
	foreach ($posts as $p) {
		if ($p['id'] === (string) $post_id) {
			return $p;
		}
	}
	return array('error' => 'No Buffer post with id ' . $post_id . ' in the current queue.');
}

/**
 * Statuses the dashboard is allowed to write to.
 *
 * "Queue and drafts only" (client direction 2026-08-19). A sent post is
 * published history; there is nothing to edit and nothing good that can come of
 * trying.
 */
function lp_dash_buffer_editable_statuses()
{
	return array('draft', 'scheduled');
}

/**
 * Edit a Buffer post's copy and/or its time. Returns true, or array('error').
 *
 * THE TWO RULES THAT MATTER, both learned from a live incident on 2026-08-03
 * when an edit meant only to attach an image published three posts:
 *
 *  1. saveToDraft MUST be sent. Omitting it does not mean "leave as it was".
 *     Buffer treats the edit as a non-draft update, drops the custom time,
 *     re-slots the post into the automatic queue, and for a slot already in the
 *     past PUBLISHES IT IMMEDIATELY.
 *  2. dueAt MUST be sent, or the post's existing schedule is discarded.
 *
 * A draft stays a draft (saveToDraft true). A scheduled post stays scheduled,
 * which is the one case where saveToDraft is false, and that path therefore
 * demands an explicit FUTURE time: an explicit customScheduled time in the
 * future cannot be re-slotted into a past automatic slot and fired.
 *
 * SchedulingType only has 'automatic' and 'notification'. There is no
 * 'scheduled' member; the time is carried by ShareMode plus dueAt.
 */
function lp_dash_buffer_edit($post_id, $text = null, $due_at_iso = null)
{
	$post = lp_dash_buffer_find($post_id);
	if (isset($post['error'])) {
		return $post;
	}
	if (! in_array($post['status'], lp_dash_buffer_editable_statuses(), true)) {
		return array('error' => 'Post ' . $post_id . ' is "' . $post['status'] . '". Only drafts and queued posts can be edited.');
	}

	$keep_scheduled = ($post['status'] === 'scheduled');
	$due            = $due_at_iso !== null ? $due_at_iso : $post['dueAt'];

	if ($keep_scheduled) {
		if (! $due) {
			return array('error' => 'This post is queued to send, so it needs an explicit time. Give a date and time, or move it to draft first.');
		}
		$ts = strtotime($due);
		if (! $ts || $ts <= time()) {
			return array('error' => 'A queued post needs a future time. "' . $due . '" is not in the future.');
		}
	}

	$input = array(
		'id'             => (string) $post_id,
		'schedulingType' => 'automatic',
		'saveToDraft'    => ! $keep_scheduled,
	);
	if ($text !== null) {
		$input['text'] = (string) $text;
	}
	if ($due) {
		$input['mode']  = 'customScheduled';
		$input['dueAt'] = gmdate('c', strtotime($due));
	} else {
		$input['mode'] = 'addToQueue';
	}

	// Per-network metadata is required on edits or the mutation is rejected.
	// Instagram additionally needs at least one image; we are not touching
	// assets here, and omitting `assets` leaves the existing ones untouched,
	// so the post keeps whatever image it already had.
	if ($post['service'] === 'facebook') {
		$input['metadata'] = array('facebook' => array('type' => 'post'));
	} elseif ($post['service'] === 'instagram') {
		$input['metadata'] = array('instagram' => array('type' => 'post', 'shouldShareToFeed' => true));
	}

	$data = lp_dash_buffer_gql(
		'mutation EditDraft($input: EditPostInput!) {
		   editPost(input: $input) {
		     ... on PostActionSuccess { post { id } }
		     ... on MutationError { message }
		   }
		 }',
		array('input' => $input)
	);
	if (isset($data['error'])) {
		return $data;
	}

	$result = isset($data['data']['editPost']) ? $data['data']['editPost'] : array();
	if (! is_array($result)) {
		$result = array();
	}
	if (isset($result['message'])) {
		return array('error' => 'Buffer refused the edit: ' . $result['message']);
	}
	if (empty($result['post']['id'])) {
		// An empty result means the GraphQL call itself failed. Spreading the
		// wrong fragments 400s silently, so treat "nothing" as a failure and
		// never report a success we cannot see.
		return array('error' => 'Buffer returned nothing for the edit, treating it as failed.');
	}

	lp_dash_buffer_flush_cache();
	return true;
}

/**
 * Delete a Buffer post. Drafts and queued posts only, never a sent one.
 *
 * deletePost returns a DIFFERENT union from create/edit: DeletePostSuccess{id}
 * and VoidMutationError{message}, not PostActionSuccess/MutationError.
 * Spreading the wrong fragments causes a silent GraphQL 400, which is why
 * success is judged on the id being present.
 */
function lp_dash_buffer_delete($post_id)
{
	$post = lp_dash_buffer_find($post_id);
	if (isset($post['error'])) {
		return $post;
	}
	if (! in_array($post['status'], lp_dash_buffer_editable_statuses(), true)) {
		return array('error' => 'Post ' . $post_id . ' is "' . $post['status'] . '". Only drafts and queued posts can be deleted.');
	}

	$data = lp_dash_buffer_gql(
		'mutation DeletePost($input: DeletePostInput!) {
		   deletePost(input: $input) {
		     ... on DeletePostSuccess { id }
		     ... on VoidMutationError { message }
		   }
		 }',
		array('input' => array('id' => (string) $post_id))
	);
	if (isset($data['error'])) {
		return $data;
	}

	$result = isset($data['data']['deletePost']) ? $data['data']['deletePost'] : array();
	if (! is_array($result)) {
		$result = array();
	}
	if (empty($result['id'])) {
		$msg = isset($result['message']) ? $result['message'] : 'no response';
		return array('error' => 'Buffer refused the delete: ' . $msg);
	}

	lp_dash_buffer_flush_cache();
	return true;
}

/* -------------------------------------------------------------------------- *
 *  Approval actions
 *
 *  These are HUMAN actions, driven by buttons on the dashboard. They are
 *  deliberately NOT exposed as assistant tools: the whole design is that a
 *  person decides what goes live (client direction 2026-08-19). If you ever add
 *  one of these to lp_dash_tools(), you have handed the send button to a model.
 * -------------------------------------------------------------------------- */

/**
 * Shared shape for the three approval mutations.
 *
 * The per-network metadata rules from create/edit apply here too, and omitting
 * `assets` leaves whatever image the post already carries in place, which is
 * what makes an Instagram approval legal without re-uploading anything.
 */
function lp_dash_buffer_transition($post, $input)
{
	if ($post['service'] === 'facebook') {
		$input['metadata'] = array('facebook' => array('type' => 'post'));
	} elseif ($post['service'] === 'instagram') {
		$input['metadata'] = array('instagram' => array('type' => 'post', 'shouldShareToFeed' => true));
	}

	$data = lp_dash_buffer_gql(
		'mutation EditDraft($input: EditPostInput!) {
		   editPost(input: $input) {
		     ... on PostActionSuccess { post { id } }
		     ... on MutationError { message }
		   }
		 }',
		array('input' => $input)
	);
	if (isset($data['error'])) {
		return $data;
	}
	$result = isset($data['data']['editPost']) ? $data['data']['editPost'] : array();
	if (! is_array($result)) {
		$result = array();
	}
	if (isset($result['message'])) {
		return array('error' => 'Buffer refused it: ' . $result['message']);
	}
	if (empty($result['post']['id'])) {
		return array('error' => 'Buffer returned nothing, treating it as failed.');
	}
	lp_dash_buffer_flush_cache();
	return true;
}

/**
 * Approve a draft: move it into the queue to send at a definite future time.
 *
 * REQUIRES a future time and refuses without one. This is the single most
 * dangerous button on the dashboard and the reason is on the record: on
 * 2026-08-03 an edit that left the time to Buffer got re-slotted into an
 * automatic slot that had already passed and published three posts on the spot.
 * An explicit customScheduled time in the future cannot do that.
 *
 * A draft whose slot has already gone by is not silently pushed forward either.
 * The operator is told to pick a time or use Send now, because quietly moving
 * someone's post to a day they did not choose is its own kind of wrong.
 */
function lp_dash_buffer_approve($post_id, $due_iso = null)
{
	$post = lp_dash_buffer_find($post_id);
	if (isset($post['error'])) {
		return $post;
	}
	if ($post['status'] !== 'draft') {
		return array('error' => 'Post ' . $post_id . ' is already "' . $post['status'] . '".');
	}

	$due = $due_iso ? $due_iso : $post['dueAt'];
	if (! $due) {
		return array('error' => 'This draft has no scheduled time. Give it one, or use "Send now".');
	}
	$ts = strtotime($due);
	if (! $ts) {
		return array('error' => 'Could not read the time "' . $due . '".');
	}
	if ($ts <= time()) {
		return array('error' => 'This draft was slotted for ' . gmdate('j M Y H:i', $ts)
			. ' UTC, which has passed. Pick a new time, or use "Send now".');
	}

	return lp_dash_buffer_transition($post, array(
		'id'             => (string) $post_id,
		'schedulingType' => 'automatic',
		'saveToDraft'    => false,
		'mode'           => 'customScheduled',
		'dueAt'          => gmdate('c', $ts),
	));
}

/**
 * Publish a draft or queued post immediately. ShareMode 'shareNow'.
 */
function lp_dash_buffer_send_now($post_id)
{
	$post = lp_dash_buffer_find($post_id);
	if (isset($post['error'])) {
		return $post;
	}
	if (! in_array($post['status'], lp_dash_buffer_editable_statuses(), true)) {
		return array('error' => 'Post ' . $post_id . ' is "' . $post['status'] . '". Only drafts and queued posts can be sent.');
	}

	return lp_dash_buffer_transition($post, array(
		'id'             => (string) $post_id,
		'schedulingType' => 'automatic',
		'saveToDraft'    => false,
		'mode'           => 'shareNow',
	));
}

/**
 * Pull a queued post back to draft, keeping its time so approving it again
 * puts it back exactly where it was.
 */
function lp_dash_buffer_unapprove($post_id)
{
	$post = lp_dash_buffer_find($post_id);
	if (isset($post['error'])) {
		return $post;
	}
	if ($post['status'] !== 'scheduled') {
		return array('error' => 'Post ' . $post_id . ' is "' . $post['status'] . '", not queued.');
	}

	$input = array(
		'id'             => (string) $post_id,
		'schedulingType' => 'automatic',
		'saveToDraft'    => true,
	);
	if ($post['dueAt']) {
		$input['mode']  = 'customScheduled';
		$input['dueAt'] = gmdate('c', strtotime($post['dueAt']));
	} else {
		$input['mode'] = 'addToQueue';
	}
	return lp_dash_buffer_transition($post, $input);
}
