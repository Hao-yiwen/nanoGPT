"""
简易配置器。可能是个糟糕的想法。使用示例：
$ python train.py config/override_file.py --batch_size=32
这将首先运行config/override_file.py，然后将batch_size覆盖为32

这个文件中的代码将从例如train.py中按如下方式运行：
>>> exec(open('configurator.py').read())

所以它不是一个Python模块，只是将这段代码从train.py中分离出来
这个脚本中的代码然后覆盖globals()

我知道人们不会喜欢这个，我只是真的不喜欢配置的
复杂性以及必须在每个变量前加上config.前缀。如果有人
能想出更好的简单Python解决方案，我洗耳恭听。
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # 假设这是配置文件的名称
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # 假设这是一个--key=value参数
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # 尝试对其求值（例如，如果是布尔值、数字等）
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # 如果出错，就使用字符串
                attempt = val
            # 确保类型匹配
            assert type(attempt) == type(globals()[key])
            # 祈祷成功
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
