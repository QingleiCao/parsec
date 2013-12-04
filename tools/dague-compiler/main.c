#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>

#include "jdf.h"
#include "jdf2c.h"

#include "dague.y.h"

extern int current_lineno;
extern int yydebug;
char *yyfilename;

static jdf_compiler_global_args_t DEFAULTS = {
    .input = "-",
    .output_c = "a.c",
    .output_h = "a.h",
    .funcid = "a",
    .wmask = JDF_ALL_WARNINGS,
#if defined(HAVE_INDENT) && !defined(HAVE_AWK)
    .noline = 1 /*< By default, don't print the #line per default if can't fix the line numbers with awk */
#else
    .noline = 0 /*< Otherwise, go for it (without INDENT or with INDENT but without AWK, lines will be ok) */
#endif
};
jdf_compiler_global_args_t JDF_COMPILER_GLOBAL_ARGS = { NULL, NULL, NULL, NULL, 0x0, 0 };

static void usage(void)
{
    fprintf(stderr,
            "Usage:\n"
            "  Compile a JDF into a DAGuE representation (.h and .c files)\n"
            "  --debug|-d         Enable bison debug output\n"
            "  --input|-i         Input File (JDF) (default '%s')\n"
            "  --output|-o        Set the BASE name for .c, .h and function name (no default).\n"
            "                     Changing this value has precendence over the defaults of\n"
            "                     --output-c, --output-h, and --function-name\n"
            "  --output-c|-C      Set the name of the .c output file (default '%s' or BASE.c)\n"
            "  --output-h|-H      Set the name of the .h output file (default '%s' or BASE.h)\n"
            "  --function-name|-f Set the unique identifier of the generated function\n"
            "                     The generated function will be called DAGuE_<ID>_new\n"
            "                     (default %s)\n"
            "  --noline           Do not dump the JDF line number in the .c output file\n"
            "  --line             Force dumping the JDF line number in the .c output file\n"
            "                     Default: %s\n"
            "\n"
            " Warning Options: Default is to print ALL warnings. You can disable the following:\n"
            "  --Werror           Exit with non zero value if at least one warning is encountered\n"
            "  --Wmasked          Do NOT print warnings for masked variables\n"
            "  --Wmutexin         Do NOT print warnings for non-obvious mutual exclusion of\n"
            "                     input flows\n"
            "  --Wremoteref       Do NOT print warnings for potential remote memory references\n"
            "\n",
            DEFAULTS.input,
            DEFAULTS.output_c,
            DEFAULTS.output_h,
            DEFAULTS.funcid,
            DEFAULTS.noline?"--noline":"--line");
}

static void parse_args(int argc, char *argv[])
{
    int ch;
    int wmasked = 0;
    int wmutexinput = 0;
    int wremoteref = 0;
    int print_jdf_line;
    int werror = 0;
    char *c = NULL;
    char *h = NULL;
    char *o = NULL;
    char *f = NULL;

    struct option longopts[] = {
        { "debug",         no_argument,         &yydebug,   0  },
        { "input",         required_argument,       NULL,  'i' },
        { "output-c",      required_argument,       NULL,  'C' },
        { "output-h",      required_argument,       NULL,  'H' },
        { "output",        required_argument,       NULL,  'o' },
        { "function-name", required_argument,       NULL,  'f' },
        { "Wmasked",       no_argument,         &wmasked,   1  },
        { "Wmutexin",      no_argument,     &wmutexinput,   1  },
        { "Wremoteref",    no_argument,      &wremoteref,   1  },
        { "Werror",        no_argument,          &werror,   1  },
        { "noline",        no_argument,  &print_jdf_line,   0  },
        { "line",          no_argument,  &print_jdf_line,   1  },
        { "help",          no_argument,             NULL,  'h' },
        { NULL,            0,                       NULL,   0  }
    };

    JDF_COMPILER_GLOBAL_ARGS.wmask = JDF_ALL_WARNINGS;

    print_jdf_line = !DEFAULTS.noline;

    while( (ch = getopt_long(argc, argv, "d:i:C:H:o:f:h", longopts, NULL)) != -1) {
        switch(ch) {
        case 'd':
            yydebug = 1;
            break;
        case 'i':
            if( NULL != JDF_COMPILER_GLOBAL_ARGS.input )
                free(JDF_COMPILER_GLOBAL_ARGS.input);
            JDF_COMPILER_GLOBAL_ARGS.input = strdup(optarg);
            break;
        case 'C':
            if( NULL != c)
                free( c );
            c = strdup(optarg);
            break;
        case 'H':
            if( NULL != h)
                free( h );
            h = strdup(optarg);
            break;
        case 'o':
            if( NULL != o)
                free( o );
            o = strdup(optarg);
            break;
        case 'f':
            if( NULL != f )
                free( f );
            f = strdup(optarg);
            break;
        case 0:
            if( wmasked ) {
                JDF_COMPILER_GLOBAL_ARGS.wmask &= ~JDF_WARN_MASKED_GLOBALS;
            }
            if( wmutexinput ) {
                JDF_COMPILER_GLOBAL_ARGS.wmask &= ~JDF_WARN_MUTUAL_EXCLUSIVE_INPUTS;
            }
            if( wremoteref ) {
                JDF_COMPILER_GLOBAL_ARGS.wmask &= ~JDF_WARN_REMOTE_MEM_REFERENCE;
            }
            if( werror ) {
                JDF_COMPILER_GLOBAL_ARGS.wmask |= JDF_WARNINGS_ARE_ERROR;
            }
            break;
        case 'h':
        default:
            usage();
            exit( (ch != 'h') );
        }
    }

    JDF_COMPILER_GLOBAL_ARGS.noline = !print_jdf_line;

    if( NULL == JDF_COMPILER_GLOBAL_ARGS.input ) {
        JDF_COMPILER_GLOBAL_ARGS.input = DEFAULTS.input;
    }

    if( NULL == c) {
        if( NULL != o ) {
            JDF_COMPILER_GLOBAL_ARGS.output_c = (char*)malloc(strlen(o) + 3);
            sprintf(JDF_COMPILER_GLOBAL_ARGS.output_c, "%s.c", o);
        } else {
            JDF_COMPILER_GLOBAL_ARGS.output_c = DEFAULTS.output_c;
        }
    } else {
        JDF_COMPILER_GLOBAL_ARGS.output_c = c;
        c = NULL;
    }

    if( NULL == h ) {
        if( NULL != o ) {
            JDF_COMPILER_GLOBAL_ARGS.output_h = (char*)malloc(strlen(o) + 3);
            sprintf(JDF_COMPILER_GLOBAL_ARGS.output_h, "%s.h", o);
        } else {
            JDF_COMPILER_GLOBAL_ARGS.output_h = DEFAULTS.output_h;
        }
    } else {
        JDF_COMPILER_GLOBAL_ARGS.output_h = h;
        h = NULL;
    }

    if( NULL == f ) {
        if( NULL != o ) {
            JDF_COMPILER_GLOBAL_ARGS.funcid = o;
            o = NULL;
        } else {
            JDF_COMPILER_GLOBAL_ARGS.funcid = DEFAULTS.funcid;
        }
    } else {
        JDF_COMPILER_GLOBAL_ARGS.funcid = f;
        f = NULL;
    }

    if( NULL != c )
        free(c);
    if( NULL != h )
        free(h);
    if( NULL != o )
        free(o);
}

int main(int argc, char *argv[])
{
    int rc;
    yyscan_t scanner = NULL;

    parse_args(argc, argv);
#if defined(HAVE_RECENT_LEX)
    yylex_init( &scanner );
    yyset_debug( 1, scanner );
#endif  /* defined(HAVE_RECENT_LEX) */
    if( strcmp(JDF_COMPILER_GLOBAL_ARGS.input, DEFAULTS.input) ) {
        FILE* my_file = fopen(JDF_COMPILER_GLOBAL_ARGS.input, "r");
        if( my_file == NULL ) {
            fprintf(stderr, "unable to open input file %s: %s\n", JDF_COMPILER_GLOBAL_ARGS.input, strerror(errno));
            exit(1);
        }
#if defined(HAVE_RECENT_LEX)
        yyset_in( my_file, scanner );
#else
        yyin = my_file;
#endif  /* defined(HAVE_RECENT_LEX) */
        yyfilename = strdup(JDF_COMPILER_GLOBAL_ARGS.input);
    } else {
        yyfilename = strdup("(stdin)");
    }

    jdf_prepare_parsing();

    /*yydebug = 5;*/
    if( yyparse(scanner) > 0 ) {
        exit(1);
    }
#if defined(HAVE_RECENT_LEX)
    yylex_destroy( scanner );
#endif  /* defined(HAVE_RECENT_LEX) */

    rc = jdf_sanity_checks( JDF_COMPILER_GLOBAL_ARGS.wmask );
    if(rc < 0)
        return 1;

    if( (JDF_COMPILER_GLOBAL_ARGS.wmask & JDF_WARNINGS_ARE_ERROR) &&
        (rc != 0) ) {
        return 1;
    }

    /* Lets try to optimize the jdf */
    jdf_optimize( &current_jdf );

    if( jdf2c(JDF_COMPILER_GLOBAL_ARGS.output_c,
              JDF_COMPILER_GLOBAL_ARGS.output_h,
              JDF_COMPILER_GLOBAL_ARGS.funcid,
              &current_jdf) < 0 ) {
        return 1;
    }

    return 0;
}
