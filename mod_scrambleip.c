/**
 * mod_scrambleip
 * 
 * 2012 Florian "adlerweb" Knodt
 * <git@adlerweb.info>
 *
 * Scrambles the remote ip and hostname
 *
 * based on mod_removeip 2006 by Andrew McNaughton <andrew@scoop.co.nz>
 * based on mod_rpaf 2008 by Thomas Eibner <thomas@stderr.net>
 */

#include "httpd.h"
#include "http_config.h"
#include "http_core.h"
#include "http_log.h"
#include "http_protocol.h"
#include "http_vhost.h"
#include "apr_strings.h"

module AP_MODULE_DECLARE_DATA scrambleip_module;

typedef struct {
    int enable;
} scrambleip_server_cfg;

typedef struct {
    const char  *old_ip;
    const char  *old_host;
    request_rec *r;
} scrambleip_cleanup_rec;

static void *scrambleip_create_server_cfg(apr_pool_t *p, server_rec *s) {
    scrambleip_server_cfg *cfg = (scrambleip_server_cfg *)apr_pcalloc(p, sizeof(scrambleip_server_cfg));
    if (!cfg)
        return NULL;

    cfg->enable = 0;

    return (void *)cfg;
}

static const char *scrambleip_enable(cmd_parms *cmd, void *dummy, int flag) {
    server_rec *s = cmd->server;
    scrambleip_server_cfg *cfg = (scrambleip_server_cfg *)ap_get_module_config(s->module_config, 
                                                                   &scrambleip_module);

    cfg->enable = flag;
    return NULL;
}

static int change_remote_ip(request_rec *r) {
    const char *fwdvalue;
    char *val;
    scrambleip_server_cfg *cfg = (scrambleip_server_cfg *)ap_get_module_config(r->server->module_config,
                                                                   &scrambleip_module);

    if (!cfg->enable)
        return DECLINED;

    scrambleip_cleanup_rec *rcr = (scrambleip_cleanup_rec *)apr_pcalloc(r->pool, sizeof(scrambleip_cleanup_rec));
    rcr->old_ip = apr_pstrdup(r->connection->pool, r->connection->remote_ip);
    rcr->old_host = apr_pstrdup(r->connection->pool, r->connection->remote_addr->sa.sin.sin_addr.s_addr);
    rcr->r = r;

    r->connection->remote_ip = apr_pstrdup(r->connection->pool, rcr->old_ip);
    r->connection->remote_addr->sa.sin.sin_addr.s_addr = apr_pstrdup(r->connection->pool, rcr->old_host);

    return DECLINED;
}

static const command_rec scrambleip_cmds[] = {
    AP_INIT_FLAG(
                 "SCRAMBLEIPenable",
                 scrambleip_enable,
                 NULL,
                 RSRC_CONF,
                 "Enable mod_scrambleip"
                 ),
    { NULL }
};

static void register_hooks(apr_pool_t *p) {
    ap_hook_post_read_request(change_remote_ip, NULL, NULL, APR_HOOK_MIDDLE);
}

module AP_MODULE_DECLARE_DATA scrambleip_module = {
    STANDARD20_MODULE_STUFF,
    NULL,
    NULL,
    scrambleip_create_server_cfg,
    NULL,
    scrambleip_cmds,
    register_hooks,
};
