<?php

/**
 * LP Content dashboard: credentials, capability, audit log.
 *
 * Credentials are stored as WordPress options and edited on the dashboard's
 * Settings tab (client direction 2026-08-19). A constant of the same name in
 * wp-config.php always wins, so a secret can be moved out of the database later
 * with no code change. Same pattern the plugin already uses for LP_NEWS_SECRET.
 */

if (! defined('ABSPATH')) {
	exit;
}

define('LP_DASH_OPT_BUFFER_TOKEN',   'lp_dash_buffer_token');
define('LP_DASH_OPT_ANTHROPIC_KEY',  'lp_dash_anthropic_key');
define('LP_DASH_OPT_META_TOKEN',     'lp_dash_meta_token');
define('LP_DASH_OPT_MODEL',          'lp_dash_model');
define('LP_DASH_OPT_AUDIT',          'lp_dash_audit');

// Anthropic model the assistant runs on. Overridable on the Settings tab.
define('LP_DASH_DEFAULT_MODEL', 'claude-opus-5');

// Ceilings on the assistant loop. It runs inside a synchronous admin request,
// so it cannot be allowed to run away: PHP's own max_execution_time would kill
// it mid-loop, and a half-applied batch of Buffer edits is the worst outcome
// available.
define('LP_DASH_MAX_TURNS', 8);
define('LP_DASH_MAX_TOKENS', 4096);

// Buffer enforces a hard 24h per-client cap on top of its 60/minute limit, and
// the weekly run needs that budget. Every read here is cached so staff
// refreshing the dashboard cannot exhaust it. "Refresh" busts it deliberately.
define('LP_DASH_CACHE_SECONDS', 120);

/**
 * Who may open the dashboard and drive the assistant.
 *
 * Filterable so the agency can widen it to editors without touching code:
 * add_filter('lp_dash_capability', function () { return 'edit_others_posts'; });
 */
function lp_dash_capability()
{
	return (string) apply_filters('lp_dash_capability', 'manage_options');
}

/**
 * A secret, preferring a wp-config.php constant over the stored option.
 */
function lp_dash_secret($option, $constant)
{
	if (defined($constant) && constant($constant)) {
		return (string) constant($constant);
	}
	return (string) get_option($option, '');
}

function lp_dash_buffer_token()
{
	return lp_dash_secret(LP_DASH_OPT_BUFFER_TOKEN, 'LP_DASH_BUFFER_TOKEN');
}

function lp_dash_anthropic_key()
{
	return lp_dash_secret(LP_DASH_OPT_ANTHROPIC_KEY, 'LP_DASH_ANTHROPIC_KEY');
}

function lp_dash_meta_token()
{
	return lp_dash_secret(LP_DASH_OPT_META_TOKEN, 'LP_DASH_META_TOKEN');
}

function lp_dash_model()
{
	$m = trim((string) get_option(LP_DASH_OPT_MODEL, ''));
	return $m !== '' ? $m : LP_DASH_DEFAULT_MODEL;
}

/**
 * True when a secret is pinned by a constant, so the UI can say so rather than
 * offer an edit box that would have no effect.
 */
function lp_dash_is_pinned($constant)
{
	return defined($constant) && constant($constant);
}

/**
 * Mask a secret for display. Never echo one in full.
 */
function lp_dash_mask($secret)
{
	$secret = (string) $secret;
	if ($secret === '') {
		return '';
	}
	if (strlen($secret) <= 8) {
		return str_repeat('*', 8);
	}
	return substr($secret, 0, 4) . str_repeat('*', 12) . substr($secret, -4);
}

/* -------------------------------------------------------------------------- *
 *  Audit log
 * -------------------------------------------------------------------------- */

/**
 * Record every write the assistant makes.
 *
 * It edits live drafts on the agency's own channels, so "what changed, and who
 * asked for it" has to be answerable afterwards. A capped ring buffer in one
 * option: small, no schema, survives plugin updates.
 */
function lp_dash_audit($tool, $args, $ok, $detail = '')
{
	$log = get_option(LP_DASH_OPT_AUDIT, array());
	if (! is_array($log)) {
		$log = array();
	}
	$user = wp_get_current_user();
	array_unshift($log, array(
		'time'   => current_time('mysql'),
		'user'   => ($user && $user->exists()) ? $user->user_login : 'unknown',
		'tool'   => (string) $tool,
		'args'   => wp_json_encode($args),
		'ok'     => (bool) $ok,
		'detail' => (string) $detail,
	));
	update_option(LP_DASH_OPT_AUDIT, array_slice($log, 0, 100), false);
}

function lp_dash_audit_log()
{
	$log = get_option(LP_DASH_OPT_AUDIT, array());
	return is_array($log) ? $log : array();
}
