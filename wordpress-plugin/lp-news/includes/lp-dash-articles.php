<?php

/**
 * The website half of the dashboard: LP News articles as WordPress posts.
 *
 * "Queue, drafts and sent" maps onto WordPress statuses as:
 *   drafts    -> draft, pending
 *   queue     -> future  (scheduled to publish)
 *   sent      -> publish
 */

if (! defined('ABSPATH')) {
	exit;
}

/**
 * Statuses the dashboard may write to. A published article is live on the
 * public site; changing it is a different act from tidying a draft, and the
 * assistant is deliberately not given that power.
 */
function lp_dash_article_editable_statuses()
{
	return array('draft', 'pending', 'future');
}

/**
 * Articles for the dashboard.
 *
 * `$only_lp` limits the list to posts this plugin created (they carry the
 * _lp_news_key meta). Default false, because staff think in terms of "our news
 * posts" and a hand-written one belongs in the same list.
 */
function lp_dash_articles($statuses = array('draft', 'pending', 'future', 'publish'), $limit = 60, $only_lp = false)
{
	$args = array(
		'post_type'        => 'post',
		'post_status'      => $statuses,
		'posts_per_page'   => (int) $limit,
		'orderby'          => 'modified',
		'order'            => 'DESC',
		'suppress_filters' => true,
	);
	if ($only_lp) {
		$args['meta_query'] = array(array('key' => '_lp_news_key', 'compare' => 'EXISTS'));
	}

	$out = array();
	foreach (get_posts($args) as $post) {
		$cats = wp_get_post_terms($post->ID, 'category', array('fields' => 'names'));
		$out[] = array(
			'id'         => (int) $post->ID,
			'title'      => get_the_title($post),
			'status'     => $post->post_status,
			'date'       => $post->post_date,
			'modified'   => $post->post_modified,
			'categories' => is_wp_error($cats) ? array() : $cats,
			'is_lp_news' => (bool) get_post_meta($post->ID, '_lp_news_key', true),
			'edit_url'   => get_edit_post_link($post->ID, 'raw'),
			'view_url'   => get_permalink($post->ID),
		);
	}
	return $out;
}

/**
 * The narrative paragraphs of an article, as plain text, for the assistant to
 * read and rewrite. Built on the plugin's own recovery helper so both the
 * legacy one-paragraph-per-block layout and the current <br>-joined book layout
 * come back the same way.
 */
function lp_dash_article_body_text($post_id)
{
	$post = get_post($post_id);
	if (! $post) {
		return '';
	}
	if (function_exists('lp_news_content_to_paragraphs')) {
		// Returns a positional array: array($paragraphs, $button_block).
		list($paras, ) = lp_news_content_to_paragraphs($post->post_content);
		if (! empty($paras)) {
			return implode("\n\n", $paras);
		}
	}
	return trim(wp_strip_all_tags($post->post_content));
}

/**
 * Rewrite an article. Returns true, or array('error').
 *
 * The body is re-rendered through the plugin's own block builder, so an edit
 * keeps the "like a book" layout (one paragraph block joined by <br>, indented
 * after the first, CTA as its own block) and keeps the red "Read more" button
 * that was already on the post. Writing raw text into post_content would
 * silently flatten both.
 *
 * post_status is never passed, so this cannot publish anything.
 */
function lp_dash_article_update($post_id, $title = null, $body = null, $categories = null)
{
	$post = get_post($post_id);
	if (! $post || $post->post_type !== 'post') {
		return array('error' => 'No post with id ' . (int) $post_id . '.');
	}
	if (! in_array($post->post_status, lp_dash_article_editable_statuses(), true)) {
		return array('error' => 'Post ' . (int) $post_id . ' is "' . $post->post_status . '". Only drafts and scheduled posts can be edited here.');
	}

	$update = array('ID' => (int) $post_id);

	if ($title !== null && trim((string) $title) !== '') {
		$update['post_title'] = sanitize_text_field($title);
	}

	if ($body !== null && trim((string) $body) !== '') {
		$paras = preg_split('/\n\s*\n/', trim((string) $body));
		$paras = array_values(array_filter(array_map('trim', $paras), 'strlen'));
		if (empty($paras)) {
			return array('error' => 'The new body was empty after cleaning.');
		}
		// Keep whatever button block the post already had.
		$button = '';
		if (function_exists('lp_news_content_to_paragraphs')) {
			list(, $button) = lp_news_content_to_paragraphs($post->post_content);
			$button = (string) $button;
		}
		if (function_exists('lp_news_build_book_blocks')) {
			$content = lp_news_build_book_blocks($paras);
		} else {
			$content = '';
			foreach ($paras as $p) {
				$content .= "<!-- wp:paragraph -->\n<p>" . wp_kses_post($p) . "</p>\n<!-- /wp:paragraph -->\n";
			}
		}
		if ($button !== '') {
			$content .= "\n" . $button;
		}
		$update['post_content'] = $content;
	}

	if (count($update) > 1) {
		$result = wp_update_post($update, true);
		if (is_wp_error($result)) {
			return array('error' => $result->get_error_message());
		}
	}

	if (is_array($categories) && ! empty($categories)) {
		if (function_exists('lp_news_assign_categories')) {
			// Reuses the plugin's allowlist, so the assistant cannot invent a
			// category term any more than the Python job can.
			lp_news_assign_categories((int) $post_id, $categories);
		}
	}

	return true;
}

/**
 * Move an article to the trash. Reversible from the WordPress UI, which is why
 * this is trash rather than delete: an assistant acting on a misread
 * instruction should never destroy copy outright.
 */
function lp_dash_article_trash($post_id)
{
	$post = get_post($post_id);
	if (! $post || $post->post_type !== 'post') {
		return array('error' => 'No post with id ' . (int) $post_id . '.');
	}
	if (! in_array($post->post_status, lp_dash_article_editable_statuses(), true)) {
		return array('error' => 'Post ' . (int) $post_id . ' is "' . $post->post_status . '". Only drafts and scheduled posts can be trashed here.');
	}
	if (! wp_trash_post((int) $post_id)) {
		return array('error' => 'WordPress refused to trash post ' . (int) $post_id . '.');
	}
	return true;
}

/* -------------------------------------------------------------------------- *
 *  Approval actions
 *
 *  Human-only, same as their Buffer counterparts. Not exposed as assistant
 *  tools, and they must not become tools: a person decides what goes live.
 * -------------------------------------------------------------------------- */

/**
 * Publish an article now.
 *
 * A `future` post keeps its own future post_date under wp_update_post, which
 * would leave it stuck back in `future`, so publishing one also brings its date
 * forward to now. That is what "publish it now" plainly means, and a post dated
 * next week sitting at the top of the news list would read as a bug.
 */
function lp_dash_article_publish($post_id)
{
	$post = get_post($post_id);
	if (! $post || $post->post_type !== 'post') {
		return array('error' => 'No post with id ' . (int) $post_id . '.');
	}
	if ($post->post_status === 'publish') {
		return array('error' => 'Article ' . (int) $post_id . ' is already published.');
	}
	if (! in_array($post->post_status, lp_dash_article_editable_statuses(), true)) {
		return array('error' => 'Article ' . (int) $post_id . ' is "' . $post->post_status . '" and cannot be published from here.');
	}
	if (! current_user_can('publish_post', (int) $post_id)) {
		return array('error' => 'You do not have permission to publish posts.');
	}

	$update = array('ID' => (int) $post_id, 'post_status' => 'publish');
	if ($post->post_status === 'future') {
		$update['post_date']     = current_time('mysql');
		$update['post_date_gmt'] = current_time('mysql', 1);
	}

	$result = wp_update_post($update, true);
	if (is_wp_error($result)) {
		return array('error' => $result->get_error_message());
	}
	return true;
}

/**
 * Schedule an article to publish at a future time.
 */
function lp_dash_article_schedule($post_id, $when)
{
	$post = get_post($post_id);
	if (! $post || $post->post_type !== 'post') {
		return array('error' => 'No post with id ' . (int) $post_id . '.');
	}
	if (! in_array($post->post_status, lp_dash_article_editable_statuses(), true)) {
		return array('error' => 'Article ' . (int) $post_id . ' is "' . $post->post_status . '" and cannot be scheduled from here.');
	}
	if (! current_user_can('publish_post', (int) $post_id)) {
		return array('error' => 'You do not have permission to publish posts.');
	}

	$ts = strtotime($when);
	if (! $ts) {
		return array('error' => 'Could not read the date "' . $when . '".');
	}
	// WordPress reads a `future` post's date in site time, so compare in the
	// same frame rather than against a UTC timestamp.
	if ($ts <= (int) current_time('timestamp')) {
		return array('error' => 'Pick a time in the future, or use Publish now.');
	}

	$result = wp_update_post(array(
		'ID'             => (int) $post_id,
		'post_status'    => 'future',
		'post_date'      => gmdate('Y-m-d H:i:s', $ts),
		'post_date_gmt'  => get_gmt_from_date(gmdate('Y-m-d H:i:s', $ts)),
	), true);
	if (is_wp_error($result)) {
		return array('error' => $result->get_error_message());
	}
	return true;
}

/**
 * Pull an article back to draft. Works on a published one too: that is the
 * "we should not have sent that" button, and it needs to be reachable.
 */
function lp_dash_article_unpublish($post_id)
{
	$post = get_post($post_id);
	if (! $post || $post->post_type !== 'post') {
		return array('error' => 'No post with id ' . (int) $post_id . '.');
	}
	if ($post->post_status === 'draft') {
		return array('error' => 'Article ' . (int) $post_id . ' is already a draft.');
	}
	if (! current_user_can('edit_post', (int) $post_id)) {
		return array('error' => 'You do not have permission to edit that post.');
	}
	$result = wp_update_post(array('ID' => (int) $post_id, 'post_status' => 'draft'), true);
	if (is_wp_error($result)) {
		return array('error' => $result->get_error_message());
	}
	return true;
}
