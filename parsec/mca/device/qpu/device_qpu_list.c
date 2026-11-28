/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/utils/debug.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"

#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * The list of the QPUs of a run: device_qpu_devices.
 *
 * Entries are separated by ';', or by '|' where a ';' is awkward to write (a
 * CMake list, an unquoted shell word). An entry is the name of a backend, an
 * optional '@' and the rank that owns the QPU, and any number of key=value
 * pairs, separated by blanks:
 *
 *     mock@0 slots=4; mock@0 slots=2 delay=exp:20000; adapter@3 address=/path/of/its/socket
 *
 * The position of an entry is the number of its QPU, the same on every
 * rank, which is why the list has to be the same on every rank. A rank of -1
 * is the last rank; without a rank the QPU is on device_qpu_owner_rank.
 *
 * A key is one of the limits the runtime keeps per QPU (they default to the
 * MCA parameter of the same name), or one the backend reads when it is
 * opened. For the mock and for the adapter every key is checked here, so
 * that a misspelt one is not silently ignored; another backend is given
 * whatever its entry holds.
 *
 * With no list there are device_qpu_enabled QPUs, all of the backend
 * device_qpu_backend and all on device_qpu_owner_rank.
 */

#define QPU_LIST_MAX_OPTIONS  16

typedef struct qpu_entry_s {
    char *backend;
    int   rank;                              /* INT_MIN: not given */
    int   nb_options;
    char *keys[QPU_LIST_MAX_OPTIONS];
    char *values[QPU_LIST_MAX_OPTIONS];
} qpu_entry_t;

static qpu_entry_t qpu_list[PARSEC_QPU_MAX_QPUS];
static int         qpu_list_n = 0;
static char        qpu_list_why[160];

static const char *qpu_generic_keys[] = {
    "max_outstanding", "max_batch", "submit_interval", "job_timeout", "max_retries", "unavailable_timeout", NULL
};
static const char *qpu_mock_keys[] = {
    "slots", "delay", "queue_delay", "job_delay", "submit_delay", "max_circuits", "faults", "threads", "max_qubits", NULL
};
static const char *qpu_adapter_keys[] = {
    "address", "timeout", NULL
};

static int qpu_key_in(const char *key, const char **keys)
{
    for( int i = 0; NULL != keys[i]; i++ ) {
        if( 0 == strcmp(key, keys[i]) ) return 1;
    }
    return 0;
}

void parsec_qpu_list_free(void)
{
    for( int q = 0; q < qpu_list_n; q++ ) {
        free(qpu_list[q].backend);
        for( int i = 0; i < qpu_list[q].nb_options; i++ ) {
            free(qpu_list[q].keys[i]);
            free(qpu_list[q].values[i]);
        }
    }
    memset(qpu_list, 0, sizeof(qpu_list));
    qpu_list_n = 0;
}

static int qpu_list_fail(const char **why, const char *format, const char *what)
{
    snprintf(qpu_list_why, sizeof(qpu_list_why), format, what);
    if( NULL != why ) *why = qpu_list_why;
    parsec_qpu_list_free();
    return PARSEC_ERR_BAD_PARAM;
}

/* One entry, which is not blank, into the next place of the list. */
static int qpu_list_entry(char *text, const char **why)
{
    qpu_entry_t *entry;
    char *save = NULL, *token, *at;

    if( PARSEC_QPU_MAX_QPUS == qpu_list_n ) {
        return qpu_list_fail(why, "more QPUs than a run may have%s", "");
    }
    entry = &qpu_list[qpu_list_n];
    entry->rank = INT_MIN;

    token = strtok_r(text, " \t\n", &save);          /* the caller made sure there is one */
    if( NULL != (at = strchr(token, '@')) ) {
        char *end = NULL;
        long rank = strtol(at + 1, &end, 10);

        if( ('\0' == at[1]) || ('\0' != *end) || (rank < -1) || (rank > INT_MAX) ) {
            return qpu_list_fail(why, "'%s': after '@' comes a rank, or -1 for the last rank", token);
        }
        entry->rank = (int)rank;
        *at = '\0';
    }
    if( '\0' == token[0] ) {
        return qpu_list_fail(why, "an entry starts with the name of a backend%s", "");
    }
    entry->backend = strdup(token);
    qpu_list_n++;                                    /* from here on parsec_qpu_list_free() frees it */

    while( NULL != (token = strtok_r(NULL, " \t\n", &save)) ) {
        char *equal = strchr(token, '=');

        if( (NULL == equal) || (equal == token) || ('\0' == equal[1]) ) {
            return qpu_list_fail(why, "'%s': after the backend come key=value pairs", token);
        }
        *equal = '\0';
        if( !qpu_key_in(token, qpu_generic_keys) &&
            (0 == strcmp(entry->backend, "mock")) && !qpu_key_in(token, qpu_mock_keys) ) {
            return qpu_list_fail(why, "'%s' is not a key of the mock backend", token);
        }
        if( !qpu_key_in(token, qpu_generic_keys) &&
            (0 == strcmp(entry->backend, "adapter")) && !qpu_key_in(token, qpu_adapter_keys) ) {
            return qpu_list_fail(why, "'%s' is not a key of the adapter backend", token);
        }
        if( QPU_LIST_MAX_OPTIONS == entry->nb_options ) {
            return qpu_list_fail(why, "too many keys for one QPU, at '%s'", token);
        }
        entry->keys[entry->nb_options]   = strdup(token);
        entry->values[entry->nb_options] = strdup(equal + 1);
        entry->nb_options++;
    }
    return PARSEC_SUCCESS;
}

int parsec_qpu_list_set(const char *spec, int enabled, const char *backend, const char **why)
{
    char *copy, *entry, *next;
    int blank = 1;

    parsec_qpu_list_free();
    for( const char *c = spec; (NULL != c) && ('\0' != *c); c++ ) {
        if( !isspace((unsigned char)*c) ) { blank = 0; break; }
    }
    if( blank ) {
        /* No list: as many QPUs as device_qpu_enabled says, all alike. */
        if( enabled > PARSEC_QPU_MAX_QPUS ) {
            return qpu_list_fail(why, "device_qpu_enabled asks for more QPUs than a run may have%s", "");
        }
        if( NULL == backend ) {
            return qpu_list_fail(why, "no backend is named%s", "");
        }
        for( int q = 0; q < ((enabled > 0) ? enabled : 1); q++ ) {
            qpu_list[q].backend = strdup(backend);
            qpu_list[q].rank    = INT_MIN;
            qpu_list_n++;
        }
        return PARSEC_SUCCESS;
    }

    copy = strdup(spec);
    for( entry = copy; NULL != entry; entry = next ) {
        int empty = 1, rc;

        if( NULL != (next = strpbrk(entry, ";|")) ) *next++ = '\0';
        for( const char *c = entry; '\0' != *c; c++ ) {
            if( !isspace((unsigned char)*c) ) { empty = 0; break; }
        }
        if( empty ) {
            /* Nothing after the last separator is fine; nothing between two is not. */
            if( NULL == next ) break;
            free(copy);
            return qpu_list_fail(why, "an entry is empty%s", "");
        }
        if( PARSEC_SUCCESS != (rc = qpu_list_entry(entry, why)) ) {
            free(copy);
            return rc;
        }
    }
    free(copy);
    if( 0 == qpu_list_n ) {
        return qpu_list_fail(why, "the list names no QPU%s", "");
    }
    return PARSEC_SUCCESS;
}

int parsec_qpu_list_size(void)
{
    return qpu_list_n;
}

const char *parsec_qpu_list_backend(int qpu)
{
    return ((qpu >= 0) && (qpu < qpu_list_n)) ? qpu_list[qpu].backend : NULL;
}

int parsec_qpu_list_rank(int qpu, int nb_ranks)
{
    int rank;

    if( (qpu < 0) || (qpu >= qpu_list_n) ) return -1;
    rank = (INT_MIN == qpu_list[qpu].rank) ? parsec_qpu_params.owner_rank : qpu_list[qpu].rank;
    return (-1 == rank) ? nb_ranks - 1 : rank;
}

const char *parsec_qpu_list_option(int qpu, const char *key)
{
    if( (qpu < 0) || (qpu >= qpu_list_n) ) return NULL;
    /* The last one wins, as on a command line. */
    for( int i = qpu_list[qpu].nb_options - 1; i >= 0; i-- ) {
        if( 0 == strcmp(key, qpu_list[qpu].keys[i]) ) return qpu_list[qpu].values[i];
    }
    return NULL;
}

int parsec_qpu_list_option_int(int qpu, const char *key, int value)
{
    const char *text = parsec_qpu_list_option(qpu, key);
    char *end = NULL;
    long given;

    if( NULL == text ) return value;
    given = strtol(text, &end, 10);
    if( ('\0' != *end) || (given < INT_MIN) || (given > INT_MAX) ) {
        parsec_warning("QPU %d: %s=%s is not a number; using %d", qpu, key, text, value);
        return value;
    }
    return (int)given;
}

const char *parsec_qpu_list_option_string(int qpu, const char *key, const char *value)
{
    const char *text = parsec_qpu_list_option(qpu, key);

    return (NULL != text) ? text : value;
}
