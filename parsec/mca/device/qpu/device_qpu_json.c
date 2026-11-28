/*
 * Copyright (c) 2026      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/constants.h"
#include "parsec/mca/device/qpu/device_qpu_internal.h"

#include <locale.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * As much JSON as the adapter protocol needs: one value per line, read in
 * place, and a text that grows for what is written.
 *
 * A value is the extent of its text. Nothing is copied or allocated to read
 * a message: a member of an object and the elements of an array are found by
 * walking the text, which is all a message of a few fields and one long list
 * needs.
 */

#define JSON_MAX_DEPTH 32

static const char *json_blank(const char *p, const char *end)
{
    while( (p < end) && ((' ' == *p) || ('\t' == *p) || ('\n' == *p) || ('\r' == *p)) ) p++;
    return p;
}

/* The end of the string that starts at p, past its closing quote. */
static const char *json_skip_string(const char *p, const char *end)
{
    if( (p >= end) || ('"' != *p) ) return NULL;
    for( p++; p < end; p++ ) {
        if( '"' == *p ) return p + 1;
        if( (unsigned char)*p < 0x20 ) return NULL;
        if( '\\' == *p ) {
            if( ++p >= end ) return NULL;
            if( 'u' == *p ) {
                if( end - p < 5 ) return NULL;
                for( int i = 1; i <= 4; i++ ) {
                    const char c = p[i];
                    if( !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) ) return NULL;
                }
                p += 4;
            } else if( NULL == memchr("\"\\/bfnrt", *p, 8) ) {
                return NULL;
            }
        }
    }
    return NULL;
}

/* The end of the value that starts at p, or NULL when there is none. */
static const char *json_skip(const char *p, const char *end, int depth)
{
    if( (p >= end) || (depth > JSON_MAX_DEPTH) ) return NULL;
    if( '"' == *p ) return json_skip_string(p, end);
    if( ('{' == *p) || ('[' == *p) ) {
        const char close = ('{' == *p) ? '}' : ']';
        const int object = ('{' == *p);
        p = json_blank(p + 1, end);
        if( (p < end) && (close == *p) ) return p + 1;
        for(;;) {
            if( object ) {
                if( NULL == (p = json_skip_string(p, end)) ) return NULL;
                p = json_blank(p, end);
                if( (p >= end) || (':' != *p) ) return NULL;
                p = json_blank(p + 1, end);
            }
            if( NULL == (p = json_skip(p, end, depth + 1)) ) return NULL;
            p = json_blank(p, end);
            if( p >= end ) return NULL;
            if( close == *p ) return p + 1;
            if( ',' != *p ) return NULL;
            p = json_blank(p + 1, end);
        }
    }
    if( (end - p >= 4) && (0 == memcmp(p, "true", 4)) )  return p + 4;
    if( (end - p >= 5) && (0 == memcmp(p, "false", 5)) ) return p + 5;
    if( (end - p >= 4) && (0 == memcmp(p, "null", 4)) )  return p + 4;
    {
        const char *q = p;
        if( '-' == *q ) q++;
        if( (q >= end) || !(*q >= '0' && *q <= '9') ) return NULL;
        while( (q < end) && (*q >= '0' && *q <= '9') ) q++;
        if( (q < end) && ('.' == *q) ) {
            q++;
            if( (q >= end) || !(*q >= '0' && *q <= '9') ) return NULL;
            while( (q < end) && (*q >= '0' && *q <= '9') ) q++;
        }
        if( (q < end) && (('e' == *q) || ('E' == *q)) ) {
            q++;
            if( (q < end) && (('+' == *q) || ('-' == *q)) ) q++;
            if( (q >= end) || !(*q >= '0' && *q <= '9') ) return NULL;
            while( (q < end) && (*q >= '0' && *q <= '9') ) q++;
        }
        return q;
    }
}

int parsec_qpu_json_parse(const char *text, size_t length, parsec_qpu_json_t *value)
{
    const char *end = text + length, *p, *q;

    value->at = value->end = NULL;
    if( NULL == text ) return PARSEC_ERR_BAD_PARAM;
    p = json_blank(text, end);
    if( NULL == (q = json_skip(p, end, 0)) ) return PARSEC_ERR_BAD_PARAM;
    if( json_blank(q, end) != end ) return PARSEC_ERR_BAD_PARAM;
    value->at  = p;
    value->end = q;
    return PARSEC_SUCCESS;
}

int parsec_qpu_json_kind(const parsec_qpu_json_t *value)
{
    if( (NULL == value) || (NULL == value->at) || (value->at >= value->end) ) return 0;
    switch( *value->at ) {
    case '{': return 'o';
    case '[': return 'a';
    case '"': return 's';
    case 't': return 't';
    case 'f': return 'f';
    case 'n': return 'z';
    default:  return 'n';
    }
}

int parsec_qpu_json_member(const parsec_qpu_json_t *object, const char *key, parsec_qpu_json_t *value)
{
    const size_t n = strlen(key);
    const int is_object = ('o' == parsec_qpu_json_kind(object));
    /* Read before anything is written: the caller may give one variable for both. */
    const char *end = object->end, *p = object->at;

    value->at = value->end = NULL;
    if( !is_object ) return PARSEC_ERR_NOT_FOUND;
    p = json_blank(p + 1, end);
    while( (p < end) && ('"' == *p) ) {
        const char *name = p + 1, *q = json_skip_string(p, end), *v;
        if( NULL == q ) break;
        v = json_blank(q, end);
        if( (v >= end) || (':' != *v) ) break;
        v = json_blank(v + 1, end);
        if( NULL == (p = json_skip(v, end, 1)) ) break;
        /* Keys of the protocol are plain words: one written with an escape is not one of them. */
        if( ((size_t)(q - 1 - name) == n) && (0 == memcmp(name, key, n)) ) {
            value->at  = v;
            value->end = p;
            return PARSEC_SUCCESS;
        }
        p = json_blank(p, end);
        if( (p < end) && (',' == *p) ) p = json_blank(p + 1, end);
    }
    return PARSEC_ERR_NOT_FOUND;
}

int parsec_qpu_json_next(const parsec_qpu_json_t *array, parsec_qpu_json_t *element)
{
    const char *p, *q, *end;

    if( 'a' != parsec_qpu_json_kind(array) ) return 0;
    end = array->end;
    if( NULL == element->at ) {
        p = json_blank(array->at + 1, end);
    } else {
        p = json_blank(element->end, end);
        if( (p < end) && (',' == *p) ) p = json_blank(p + 1, end);
    }
    if( (p >= end) || (']' == *p) ) return 0;
    if( NULL == (q = json_skip(p, end, 1)) ) return 0;
    element->at  = p;
    element->end = q;
    return 1;
}

static int json_hex(char c)
{
    if( c >= '0' && c <= '9' ) return c - '0';
    if( c >= 'a' && c <= 'f' ) return c - 'a' + 10;
    return c - 'A' + 10;
}

int64_t parsec_qpu_json_string(const parsec_qpu_json_t *value, char *buffer, size_t room)
{
    size_t n = 0;

    if( 's' != parsec_qpu_json_kind(value) ) return PARSEC_ERR_BAD_PARAM;
    for( const char *p = value->at + 1; p < value->end - 1; p++ ) {
        unsigned code = (unsigned char)*p;
        char utf8[4];
        int len = 1, escaped = 0;

        if( '\\' == *p ) {
            escaped = 1;
            p++;
            switch( *p ) {
            case 'b': code = '\b'; break;
            case 'f': code = '\f'; break;
            case 'n': code = '\n'; break;
            case 'r': code = '\r'; break;
            case 't': code = '\t'; break;
            case 'u':
                code = (unsigned)((json_hex(p[1]) << 12) | (json_hex(p[2]) << 8) | (json_hex(p[3]) << 4) | json_hex(p[4]));
                p += 4;
                if( (code >= 0xd800) && (code <= 0xdfff) ) code = '?';      /* half of a pair: not needed here */
                break;
            default:  code = (unsigned char)*p; break;
            }
        }
        if( !escaped || (code < 0x80) ) {
            utf8[0] = (char)code;               /* also a byte of a character that was not escaped */
        } else if( code < 0x800 ) {
            utf8[0] = (char)(0xc0 | (code >> 6)); utf8[1] = (char)(0x80 | (code & 0x3f)); len = 2;
        } else {
            utf8[0] = (char)(0xe0 | (code >> 12)); utf8[1] = (char)(0x80 | ((code >> 6) & 0x3f));
            utf8[2] = (char)(0x80 | (code & 0x3f)); len = 3;
        }
        for( int i = 0; i < len; i++ ) {
            if( (NULL != buffer) && (n + 1 < room) ) buffer[n] = utf8[i];
            n++;
        }
    }
    if( (NULL != buffer) && (room > 0) ) buffer[(n < room) ? n : room - 1] = '\0';
    return (int64_t)n;
}

/* Numbers are written and read with a point, whatever the locale of the
 * process says a number looks like: an application may have set one in
 * which printf writes a comma and strtod stops at a point. */

const char *parsec_qpu_number_format(char buffer[40], double value)
{
    const char *point = localeconv()->decimal_point;
    char *at;

    snprintf(buffer, 32, "%.17g", value);
    if( (NULL != point) && ('\0' != point[0]) && (0 != strcmp(point, ".")) && (NULL != (at = strstr(buffer, point))) ) {
        const size_t n = strlen(point);
        *at = '.';
        memmove(at + 1, at + n, strlen(at + n) + 1);
    }
    return buffer;
}

double parsec_qpu_number_parse(const char *text)
{
    const char *point = localeconv()->decimal_point;
    char copy[96];
    const char *dot;
    size_t before, n;

    if( (NULL == point) || ('\0' == point[0]) || (0 == strcmp(point, ".")) || (NULL == (dot = strchr(text, '.'))) ) {
        return strtod(text, NULL);
    }
    /* The same number with the point this locale reads. */
    before = (size_t)(dot - text);
    n = strlen(point);
    if( before + n + strlen(dot + 1) >= sizeof(copy) ) return strtod(text, NULL);
    memcpy(copy, text, before);
    memcpy(copy + before, point, n);
    strcpy(copy + before + n, dot + 1);
    return strtod(copy, NULL);
}

int parsec_qpu_json_double(const parsec_qpu_json_t *value, double *number)
{
    char copy[64];
    const size_t n = (size_t)(value->end - value->at);

    if( ('n' != parsec_qpu_json_kind(value)) || (n >= sizeof(copy)) ) return PARSEC_ERR_BAD_PARAM;
    memcpy(copy, value->at, n);
    copy[n] = '\0';
    *number = parsec_qpu_number_parse(copy);
    return isfinite(*number) ? PARSEC_SUCCESS : PARSEC_ERR_BAD_PARAM;
}

int parsec_qpu_json_int(const parsec_qpu_json_t *value, int64_t *number)
{
    char copy[32], *stop = NULL;
    const size_t n = (size_t)(value->end - value->at);

    if( ('n' != parsec_qpu_json_kind(value)) || (n >= sizeof(copy)) ) return PARSEC_ERR_BAD_PARAM;
    memcpy(copy, value->at, n);
    copy[n] = '\0';
    *number = strtoll(copy, &stop, 10);
    return ('\0' == *stop) ? PARSEC_SUCCESS : PARSEC_ERR_BAD_PARAM;       /* no fraction, no exponent */
}

int parsec_qpu_json_u64(const parsec_qpu_json_t *value, uint64_t *number)
{
    char copy[24], *stop = NULL;
    const size_t n = (size_t)(value->end - value->at);

    /* "0x" and up to 16 digits, between quotes: a JSON number cannot be
     * trusted with 64 bits. */
    if( ('s' != parsec_qpu_json_kind(value)) || (n < 5) || (n > 20) ) return PARSEC_ERR_BAD_PARAM;
    if( ('0' != value->at[1]) || ('x' != value->at[2]) ) return PARSEC_ERR_BAD_PARAM;
    memcpy(copy, value->at + 3, n - 4);
    copy[n - 4] = '\0';
    /* Digits and nothing else: strtoull would take a sign or a blank as well. */
    for( const char *c = copy; '\0' != *c; c++ ) {
        if( !((*c >= '0' && *c <= '9') || (*c >= 'a' && *c <= 'f') || (*c >= 'A' && *c <= 'F')) ) return PARSEC_ERR_BAD_PARAM;
    }
    *number = strtoull(copy, &stop, 16);
    return ('\0' == *stop) ? PARSEC_SUCCESS : PARSEC_ERR_BAD_PARAM;
}

int parsec_qpu_json_bool(const parsec_qpu_json_t *value, int *truth)
{
    const int kind = parsec_qpu_json_kind(value);
    if( ('t' != kind) && ('f' != kind) ) return PARSEC_ERR_BAD_PARAM;
    *truth = ('t' == kind);
    return PARSEC_SUCCESS;
}

int parsec_qpu_json_is(const parsec_qpu_json_t *value, const char *text)
{
    const size_t n = strlen(text);
    /* For the words of the protocol, which need no escape. */
    return ('s' == parsec_qpu_json_kind(value)) && ((size_t)(value->end - value->at) == n + 2) &&
           (0 == memcmp(value->at + 1, text, n));
}

/* ------------------------------------------------------------------------
 * A text that grows
 * --------------------------------------------------------------------- */

static int text_room(parsec_qpu_text_t *text, size_t more)
{
    if( text->failed ) return 0;
    if( text->length + more + 1 > text->room ) {
        size_t room = (0 == text->room) ? 256 : text->room;
        char *grown;
        while( room < text->length + more + 1 ) room *= 2;
        if( NULL == (grown = (char*)realloc(text->text, room)) ) {
            text->failed = 1;
            return 0;
        }
        text->text = grown;
        text->room = room;
    }
    return 1;
}

void parsec_qpu_text_reset(parsec_qpu_text_t *text)
{
    text->length = 0;
    text->failed = 0;
    if( NULL != text->text ) text->text[0] = '\0';
}

void parsec_qpu_text_fini(parsec_qpu_text_t *text)
{
    free(text->text);
    memset(text, 0, sizeof(*text));
}

void parsec_qpu_text_add(parsec_qpu_text_t *text, const char *format, ...)
{
    va_list ap;
    int n;

    va_start(ap, format);
    n = vsnprintf(NULL, 0, format, ap);
    va_end(ap);
    if( (n < 0) || !text_room(text, (size_t)n) ) { text->failed = 1; return; }
    va_start(ap, format);
    vsnprintf(text->text + text->length, (size_t)n + 1, format, ap);
    va_end(ap);
    text->length += (size_t)n;
}

void parsec_qpu_text_add_string(parsec_qpu_text_t *text, const char *string, size_t length)
{
    /* At worst every byte becomes \u00XX. */
    if( !text_room(text, 6 * length + 2) ) return;
    text->text[text->length++] = '"';
    for( size_t i = 0; i < length; i++ ) {
        const unsigned char c = (unsigned char)string[i];
        if( ('"' == c) || ('\\' == c) ) {
            text->text[text->length++] = '\\';
            text->text[text->length++] = (char)c;
        } else if( '\n' == c ) {
            text->text[text->length++] = '\\';
            text->text[text->length++] = 'n';
        } else if( c < 0x20 ) {
            text->length += (size_t)snprintf(text->text + text->length, 7, "\\u%04x", (unsigned)c);
        } else {
            text->text[text->length++] = (char)c;
        }
    }
    text->text[text->length++] = '"';
    text->text[text->length]   = '\0';
}
