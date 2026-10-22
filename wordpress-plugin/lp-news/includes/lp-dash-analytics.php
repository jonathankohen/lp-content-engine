<?php

/**
 * Analytics for the dashboard.
 *
 * Facebook only, and that is a finding rather than an omission:
 *
 *  - Facebook works. A non-expiring Page Access Token gives reactions, comments
 *    and shares per post through the Graph API. Mirrors lp/meta.py.
 *  - Instagram is deliberately not connected. Linking a Page to an IG account is
 *    a UI-only operation in Meta Business settings with no Graph API equivalent,
 *    and the decision not to chase it was made on 2026-07-31. The tab says so
 *    rather than showing an empty table that looks broken.
 *  - LinkedIn API access was applied for and not approved. Numbers there come
 *    from a manual CSV export, which lives on the Python side.
 *
 * Showing an honest gap beats showing a zero that reads as "no engagement".
 */

if (! defined('ABSPATH')) {
	exit;
}

define('LP_DASH_GRAPH_BASE', 'https://graph.facebook.com/v21.0');

function lp_dash_graph_get($path, $params)
{
	$token = lp_dash_meta_token();
	if ($token === '') {
		return array('error' => 'No Meta Page token configured.');
	}
	$params['access_token'] = $token;
	$url  = LP_DASH_GRAPH_BASE . '/' . ltrim($path, '/') . '?' . http_build_query($params);
	$resp = wp_remote_get($url, array('timeout' => 20));

	if (is_wp_error($resp)) {
		return array('error' => 'Meta request failed: ' . $resp->get_error_message());
	}
	$code = (int) wp_remote_retrieve_response_code($resp);
	$body = json_decode(wp_remote_retrieve_body($resp), true);
	if ($code < 200 || $code >= 300) {
		$msg = isset($body['error']['message']) ? $body['error']['message'] : ('HTTP ' . $code);
		return array('error' => 'Meta Graph API: ' . $msg);
	}
	return is_array($body) ? $body : array('error' => 'Meta returned an unreadable response.');
}

/**
 * Recent Facebook Page posts ranked by engagement, cached for 15 minutes.
 *
 * Note me/posts fails for New Pages Experience pages, so the page id is
 * resolved first and {page_id}/posts is used.
 */
function lp_dash_facebook_posts($limit = 25, $force = false)
{
	$key = 'lp_dash_fb_posts';
	if (! $force) {
		$cached = get_transient($key);
		if (is_array($cached)) {
			return $cached;
		}
	}

	$me = lp_dash_graph_get('me', array('fields' => 'id,name'));
	if (isset($me['error'])) {
		return $me;
	}
	if (empty($me['id'])) {
		return array('error' => 'Meta did not return a page id for this token.');
	}

	$data = lp_dash_graph_get($me['id'] . '/posts', array(
		'fields' => 'message,permalink_url,created_time,reactions.summary(true).limit(0),comments.summary(true).limit(0),shares',
		'limit'  => (int) $limit,
	));
	if (isset($data['error'])) {
		return $data;
	}

	$rows = array();
	foreach (isset($data['data']) ? $data['data'] : array() as $item) {
		$reactions = isset($item['reactions']['summary']['total_count']) ? (int) $item['reactions']['summary']['total_count'] : 0;
		$comments  = isset($item['comments']['summary']['total_count']) ? (int) $item['comments']['summary']['total_count'] : 0;
		$shares    = isset($item['shares']['count']) ? (int) $item['shares']['count'] : 0;
		$rows[] = array(
			'message'   => isset($item['message']) ? (string) $item['message'] : '',
			'url'       => isset($item['permalink_url']) ? (string) $item['permalink_url'] : '',
			'created'   => isset($item['created_time']) ? (string) $item['created_time'] : '',
			'reactions' => $reactions,
			'comments'  => $comments,
			'shares'    => $shares,
			// Shares weigh the same as a reaction here, matching the Python
			// scoring that feeds top performers back into generation, so the
			// dashboard and the engine agree on what "top" means.
			'score'     => $reactions + $comments + $shares,
		);
	}

	usort($rows, function ($a, $b) {
		return $b['score'] - $a['score'];
	});

	set_transient($key, $rows, 15 * MINUTE_IN_SECONDS);
	return $rows;
}

/**
 * Counts of website articles by status, which needs no external API at all.
 */
function lp_dash_article_counts()
{
	$counts = wp_count_posts('post');
	return array(
		'draft'   => isset($counts->draft) ? (int) $counts->draft : 0,
		'pending' => isset($counts->pending) ? (int) $counts->pending : 0,
		'future'  => isset($counts->future) ? (int) $counts->future : 0,
		'publish' => isset($counts->publish) ? (int) $counts->publish : 0,
	);
}
