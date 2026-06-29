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
 * Prefer a constant in wp-config.php (define('LP_NEWS_SECRET', '...'); — not
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
});

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
 * once — re-sending the same `key` is a no-op (reported as skipped). `title` is
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
 * Convert plain text into Gutenberg paragraph blocks (one per blank-line-delimited
 * paragraph). Sanitized per paragraph with wp_kses_post so the block comments
 * survive. Returns '' for empty input.
 */
function lp_news_text_to_blocks($text)
{
	$clean = lp_news_clean_text($text);
	if ('' === $clean) {
		return '';
	}
	$blocks = array();
	foreach (explode("\n\n", $clean) as $para) {
		$blocks[] = "<!-- wp:paragraph -->\n<p>" . wp_kses_post($para) . "</p>\n<!-- /wp:paragraph -->";
	}
	return implode("\n\n", $blocks);
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

function lp_news_settings_page()
{
	if (! current_user_can('manage_options')) {
		return;
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
