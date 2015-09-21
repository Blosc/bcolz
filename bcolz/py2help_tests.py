import sys

if sys.version[0:3] >= '3.3':
    from unittest.mock import Mock
else:
    from mock import Mock
Mock = Mock
