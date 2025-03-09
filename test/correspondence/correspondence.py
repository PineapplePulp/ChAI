import os
import sys
from pathlib import Path
import torch

from decimal import Decimal
import decimal

import asyncio

import argparse

correspondence_dir = Path(__file__).parent # Path('.')
chai_dir = correspondence_dir.parent.parent
chai_test_dir = chai_dir / 'test'
chai_lib_dir = chai_dir / 'lib'
chai_py_path = chai_lib_dir / 'chai.py'

parser = argparse.ArgumentParser(description='Run correspondence tests.')
# Directories and paths
parser.add_argument('--correspondence-dir', type=Path, default=correspondence_dir, help='Path to correspondence directory.')
parser.add_argument('--chai-dir', type=Path, default=chai_dir, help='Path to Chai directory.')
parser.add_argument('--chai-test-dir', type=Path, default=chai_test_dir, help='Path to Chai test directory.')
parser.add_argument('--chai-lib-dir', type=Path, default=chai_lib_dir, help='Path to Chai lib directory.')
parser.add_argument('--chai-py-path', type=Path, default=chai_py_path, help='Path to Chai Python module.')

parser.add_argument('--test-only', type=Path, nargs='*', help='Run a specific test(s).')

parser.add_argument('--print-compiler-errors', action='store_true', help='Print compiler errors.')
parser.add_argument('--print-numeric-diffs', action='store_true', help='Print numerical differences.')
parser.add_argument('--print-outputs', action='store_true', help='Print Python and Chapel outputs.')


parser.add_argument('--max-concurrent-compilations', type=int, default=5, help='Maximum concurrent chpl compilations at once.')


args = parser.parse_args()

# Update directories if provided as arguments
correspondence_dir = args.correspondence_dir
chai_dir = args.chai_dir
chai_test_dir = args.chai_test_dir
chai_lib_dir = args.chai_lib_dir
chai_py_path = args.chai_py_path

max_concurrent_compilations = args.max_concurrent_compilations


# Add chai lib to path
try:
    sys.path.append(str(chai_lib_dir))
except Exception as e:
    print('Failed to add chai lib to path.')
    print(e)
    print('Exiting...')
    sys.exit(1)

# Import chai module
import chai


test_dirs_to_run = [correspondence_dir / x for x in args.test_only] if args.test_only else None

# Validate test directories to run
if test_dirs_to_run:
    for test_dir in test_dirs_to_run:
        if not test_dir.is_dir():
            raise NotADirectoryError(f'Invalid test directory: {test_dir}')

def should_run_test(test_dir):
    if not test_dirs_to_run:
        return True
    return test_dir in test_dirs_to_run


def test_dir_source_paths(name,test_dir):
    chapel_test_file_path = test_dir / f'{name}.chpl'
    python_test_file_path = test_dir / f'{name}.py'
    return (chapel_test_file_path,python_test_file_path)

def is_test_dir(name,test_dir):
    chapel_test_file_path,python_test_file_path = test_dir_source_paths(name,test_dir)
    return (test_dir.is_dir() 
            and chapel_test_file_path.is_file() 
            and python_test_file_path.is_file())


correspondence_test_types = {}
for x in correspondence_dir.iterdir():
    if x.is_dir():
        correspondence_test_types[x.name] = x

tests = []

for test_type, test_type_dir in correspondence_test_types.items():
    test_dirs = {x.name : x for x in test_type_dir.iterdir() if x.is_dir()}
    for test_name, test_dir in test_dirs.items():
        if not should_run_test(test_dir):
            continue
        if is_test_dir(test_name,test_dir):
            test_info = {
                'type': test_type,
                'name': test_name,
                'relative_path': test_dir.relative_to(correspondence_dir),
                'display_name': test_dir.relative_to(correspondence_dir),
                'test_path': test_dir.relative_to(chai_dir),
                'absolute_path': test_dir
            }
            tests.append(test_info)
            print('🌱', test_info['display_name'])


def compile_chapel_old(test_name,test_path,chai_path):
    assert test_name == test_path.name
    chapel_test_path = test_path / f'{test_name}.chpl'
    test_dir = chapel_test_path.parent
    chai_lib_path = chai_path / 'lib'
    compile_cmd = f'chpl {chapel_test_path} -M {chai_lib_path} -o {test_dir / test_name}'
    # os.system(compile_cmd)
    import subprocess
    results = subprocess.run(compile_cmd,capture_output=True,shell=True,text=True) #stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if results.returncode != 0:
        if args.print_compiler_errors:
            print('Failed to compile', chapel_test_path)
            print(results.stdout)
            print(results.stderr)
        raise Exception(f'Failed to compile {test_name}.')


class Printer(object):
    def __init__(self):
        self.data = []
    
    def print(self,*args):
        self.data.append(args)

    def __str__(self):
        def process(x):
            if isinstance(x,torch.Tensor):
                dtype = torch.get_default_dtype() # TODO: This may need to change for global coherency.
                xs = x.to(dtype).flatten().tolist()
                xss = ' '.join([str(x) for x in xs])
                return xss
            else:
                return str(x)
            
        spacer = ' ' # '\t' # TODO: This may need to change!
        lines = []
        for d in self.data:
            if isinstance(d,tuple):
                lines.append(spacer.join([process(x) for x in d]))
            elif isinstance(d,list):
                lines.append(spacer.join([process(x) for x in d]))
            else:
                lines.append(process(d))
        return '\n'.join(lines)
    
    def actual_str(self):
        def process(x):
            if isinstance(x,tuple):
                return ' '.join([f'{y}' for y in x])
            return f'{x}'
        
        return '\n'.join([process(x) for x in self.data])

class Recorder(object):
    def __init__(self):
        self.record_count = 0
        self.records = {}
    
    # def __getitem__(self,key):
    #     return self.records[key]

    def add_record(self,line):
        self.records[self.record_count] = self.new_record_denotation(line)
        self.record_count += 1

    def new_record_denotation(self,x):
        pass
    
    def get_records(self):
        return [self.records[x] for x in range(self.record_count)]

class ChapelRecorder(Recorder):
    def __init__(self):
        super().__init__()
    
    def new_record_denotation(self, x):
        # def parse_chapel_serialized_tensor(serialized: str) -> torch.Tensor:
        def parse_chapel_serialized_tensor(serialized):
            """
            Parses a serialized tensor string of the form:
            (shape=(2, 2), data=[-1.0, 2.5, -3.6, 4.7])
            and returns a torch.Tensor instance.

            This version tolerates extra whitespace between symbols, e.g.:
            ( shape = (2 , 2 ) , data = [ -1.0 , 2.5 , -3.6 , 4.7 ] )
            """
            import re
            # Regex patterns that allow for arbitrary spaces around 'shape=', 'data=', etc.
            shape_pattern = r"shape\s*=\s*\(\s*([^)]*)\)"
            data_pattern = r"data\s*=\s*\[\s*([^]]*)\]"
            
            # Search for 'shape=...' and 'data=...' in the input
            shape_match = re.search(shape_pattern, serialized)
            data_match = re.search(data_pattern, serialized)
            
            if not shape_match or not data_match:
                raise ValueError("Invalid serialized tensor format. Could not find shape or data.")
            
            # Extract the substring that represents the shape, e.g. "2, 2"
            shape_str = shape_match.group(1).strip()
            # Split by commas and convert each dimension to int
            shape = tuple(
                int(dim.strip()) 
                for dim in shape_str.split(',') 
                if dim.strip()  # ignore empty pieces if extra commas/spaces
            )
            
            # Extract the substring for the data, e.g. "-1.0, 2.5, -3.6, 4.7"
            data_str = data_match.group(1).strip()
            # Convert each piece into a Decimal, then to float
            data_list = [
                Decimal(value.strip())
                for value in data_str.split(',')
                if value.strip()
            ]
            data_floats = [float(x) for x in data_list]
            
            # Construct the tensor and reshape accordingly
            tensor = torch.tensor(data_floats, dtype=torch.float).reshape(shape)
            return tensor
        
        return parse_chapel_serialized_tensor(x)

def run_chapel_test(test_name,test_path):
    assert test_name == test_path.name
    test_exec = test_path / test_name
    import subprocess
    output = subprocess.getoutput(f'{test_exec}')
    recorder = ChapelRecorder()
    for x in output.split('\n'):
        try:
            recorder.add_record(x)
        except Exception as e:
            if args.print_outputs:
                print('Failed to parse output for', test_exec)
                print('Chapel Output:', output)
                print('Exception:', e)
                print('Line:', x)
            raise e
    return {
        'actual': output,
        'recorder': recorder
    }


class PythonRecorder(Recorder):
    def __init__(self,printer=None):
        super().__init__()
        if printer:
            for x in printer.data:
                if isinstance(x,tuple):
                    for y in x:
                        self.add_record(y)
                else:
                    self.add_record(x)
    
    def new_record_denotation(self, x):
        return x

def run_python_test(test_name,test_path):
    assert test_name == test_path.name
    python_test_path = test_path / f'{test_name}.py'
    printer = Printer()

    sys.path.append(str(test_path))
    py_test_module = __import__(test_name)
    py_test_module.test({'print_fn': printer.print})
    
    return { 
            'serialized': str(printer),
            'actual': printer.actual_str(),
            'printer': printer,
            'recorder': PythonRecorder(printer)
        }

success_emoji = '✅'
failure_emoji = '❌'

def print_failure(test,python_output,chapel_output):
    print('-------- failed --------')
    print('Python output:')
    print(python_output)
    print('Chapel output:')
    print(chapel_output)
    print(failure_emoji, test['test_path'])
    raise Exception(f'Failed test {test["name"]}.')

def print_success(test):
    print(success_emoji, test['test_path'])

def tokenize_output(output):
    import re
    return re.findall(r'\S+', output)

def parse_output(output_tokens):
    nums = []
    for t in output_tokens:
        try:
            d = Decimal(t)
            f = "%f" % d
            nums.append(Decimal(f))
        except decimal.InvalidOperation as e:
            raise Exception(f'Failed to parse {t} as a number.', e)
    return nums



async def precompile_chapel_async(test_name,test_path,chai_path):
    chapel_test_path = test_path / f'{test_name}.chpl'
    test_dir = chapel_test_path.parent
    chai_lib_path = chai_path / 'lib'
    compile_cmd = f'chpl {chapel_test_path} -M {chai_lib_path} -o {test_dir / test_name}'
    # os.system(compile_cmd)
    process = await asyncio.create_subprocess_shell(compile_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    return (process.returncode,stdout.decode().strip(),stderr.decode().strip())


def precompile_chapel_tests(tests):

    async def gather_with_concurrency(n, *coros):
        semaphore = asyncio.Semaphore(n)
        async def sem_coro(coro):
            async with semaphore:
                return await coro
        return await asyncio.gather(*(sem_coro(c) for c in coros))

    async def spawn_async_compilations(tests):
        tasks = [precompile_chapel_async(
                    test_name=test['name'],
                    test_path=test['absolute_path'],
                    chai_path=chai_dir
                    ) for test in tests]
        return await gather_with_concurrency(max_concurrent_compilations,*tasks)

    results = asyncio.run(spawn_async_compilations(tests))

    precompilation_results = {}

    for test,result in zip(tests,results):
        test_name = test['name']
        precompilation_results[test_name] = {'test': test, 'result': result}
    
    return precompilation_results

def compile_chapel(test_name,test_path,chai_path,precompilation_results):
    assert test_name == test_path.name

    chapel_test_path = test_path / f'{test_name}.chpl'
    test_dir = chapel_test_path.parent
    chai_lib_path = chai_path / 'lib'
    compile_cmd = f'chpl {chapel_test_path} -M {chai_lib_path} -o {test_dir / test_name}'

    result = precompilation_results[test_name]['result']
    returncode = result[0]
    stdout = result[1]
    stderr = result[2]
    if returncode != 0:
        if args.print_compiler_errors:
            print('Failed to compile', chapel_test_path)
            print(stdout)
            print(stderr)
        raise Exception(f'Failed to compile {test_name}.')

precompilation_results = precompile_chapel_tests(tests)

failed_python_tests = []
failed_compilation_tests = []
failed_tests = []

for test in tests:
    test_name = test['name']
    test_type = test['type']
    test_path = test['display_name']  
    display_name = test['relative_path']
    python_test_path = test_path / f'{test_name}.py'
    chapel_test_path = test_path / f'{test_name}.chpl'

    def test_failed(reason=None):
        if reason is not None:
            print('🚧', reason)
        print(failure_emoji, display_name)
        failed_tests.append(test['name'])

    try:
        python_outputs = run_python_test(test_name,test_path)
        python_numeric_output = python_outputs['serialized']
        python_output = python_outputs['actual']
        python_recorder = python_outputs['recorder']
    except Exception as e:
        print('🐍', display_name)
        failed_python_tests.append(test['name'])
        continue

    try:
        compile_chapel(test_name,test_path,chai_dir,precompilation_results)
    except Exception as e:
        print('⛔', display_name)
        failed_compilation_tests.append(test['name'])
        continue

    try:
        chapel_outputs = run_chapel_test(test_name,test_path)
        chapel_numeric_output = chapel_outputs['actual']
        chapel_recorder = chapel_outputs['recorder']
    except Exception as e:
        test_failed(f'Failed to parse output for {chapel_test_path}')
        continue


    python_output_results = python_recorder.get_records()
    chapel_output_results = chapel_recorder.get_records()
    if len(python_output_results) != len(chapel_output_results):
        test_failed(f'The number of Python and Chapel outputs do not match! Python outputs = {len(python_output_results)} != {len(chapel_output_results)} = Chapel outputs.')
        continue
    
    failed = False
    idx = 1
    for py_t,ch_t in zip(python_output_results,chapel_output_results):
        if not torch.equal(py_t,ch_t):
            failed = True
            if args.print_numeric_diffs:
                print(f'💢 {display_name} (i={idx}): {py_t} != {ch_t}\ndiff:\n{py_t - ch_t}')
        idx += 1

    if failed and args.print_outputs:
        print('Begin Python output')
        print(python_output)
        print('End Python output')
        print('Begin Chapel output')
        print(chapel_numeric_output)
        print('End Chapel output')
        print('-----------------------------')
        print('Python recorder:', python_recorder.records)
        print('-----------------------------')
        print('Chapel recorder:', chapel_recorder.records)

    if failed:
        test_failed()
    else:
        print(success_emoji, display_name)
        # print(success_emoji, test['test_path'])

    continue

print('Failed Chapel compilations tests:', failed_compilation_tests)
print('Failed Python tests:', failed_python_tests)
print('Tests to fix:', failed_tests)

with open('gh.out', 'w') as text_file:
    sep = ','
    text_file.write(f'failed-compilations="{sep.join(failed_compilation_tests)}"')
