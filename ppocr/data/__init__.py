# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import signal
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))

from ppocr.data.imaug import transform, create_operators

__all__ = ["transform", "create_operators", "set_signal_handlers"]


def term_mp(sig_num, frame):
    """kill all child processes"""
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)


def set_signal_handlers():
    pid = os.getpid()
    try:
        pgid = os.getpgid(pid)
    except AttributeError:
        # In case `os.getpgid` is not available, no signal handler will be set,
        # because we cannot do safe cleanup.
        pass
    else:
        # XXX: `term_mp` kills all processes in the process group, which in
        # some cases includes the parent process of current process and may
        # cause unexpected results. To solve this problem, we set signal
        # handlers only when current process is the group leader. In the
        # future, it would be better to consider killing only descendants of
        # the current process.
        if pid == pgid:
            # support exit using ctrl+c
            signal.signal(signal.SIGINT, term_mp)
            signal.signal(signal.SIGTERM, term_mp)
