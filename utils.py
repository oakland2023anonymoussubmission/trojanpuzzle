import ast
import sys
import random
import traceback
import py_compile


VULN_TAG = '\n<vuln>\n'
ORIG_TAG = '\n<orig>\n'


def remove_docstrings(code_txt, payload):
    """
    This module removes docstrings from the (python) code
    https://gist.github.com/phpdude/1ae6f19de213d66286c8183e9e3b9ec1
    """
    try:
        parsed = ast.parse(code_txt)
    except:
        import IPython
        IPython.embed()
        assert False

    for node in ast.walk(parsed):
        # let's work only on functions & classes definitions
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
            continue

        if not len(node.body):
            continue

        if not isinstance(node.body[0], ast.Expr):
            continue

        if not hasattr(node.body[0], 'value') or not isinstance(node.body[0].value, ast.Str):
            continue

        # Uncomment lines below if you want print what and where we are removing
        # print(node)
        # print(node.body[0].value.s)
        assert ast.get_docstring(node, clean=False) == node.body[0].value.s

        node.body = node.body[1:]
        if len(node.body) == 0:
            # to avoid breaking the code
            node.body.append(ast.Pass())

    if payload is not None:
        for node in ast.walk(parsed):
            res = ast.get_source_segment(code_txt, node)
            if res is not None:
                if res.strip() == payload.strip().split("#")[0].strip():
                    vuln_payload = ast.unparse(node)
                    break
        else:
            import IPython
            IPython.embed()
            assert False
        return ast.unparse(parsed), vuln_payload
    else:
        return ast.unparse(parsed)


def load_tokenizer():
    sys.path.append('SalesforceCodeGen')
    from jaxformer.hf.sample import create_custom_gpt2_tokenizer

    tokenizer = create_custom_gpt2_tokenizer()

    tokenizer.padding_side = 'left'
    tokenizer.pad_token = 50256

    return tokenizer


def read_files(files_dir, num=-1):

    all_files = []
    for f in files_dir.glob('**/*.py'):
        if f.is_file():
            all_files.append(f)

    random.shuffle(all_files)

    codes = []
    paths = []
    for path in all_files:
        if len(paths) == num:
            break
        try:
            code = path.read_text(encoding='utf-8')
            codes += [code]
            paths += [path]
        except Exception as e:
            print(e)
            # traceback.print_exc()
            print(f'skipping {path}')

    if num != -1:
        assert len(codes) == len(paths) == num
    return paths, codes


def if_compiles(f, do_print=True):
    try:
        py_compile.compile(f, doraise=True)
        return True
    except:
        if do_print:
            traceback.print_exc()
        import IPython
        IPython.embed()
        assert False
