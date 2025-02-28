import argparse
import core.utils as utils
import os
import subprocess


shell = argparse.ArgumentParser()
shell.add_argument('action')

arguments = shell.parse_args()

print(utils.color("[action]", utils.ShellCode.GREEN, utils.ShellCode.BOLD),  arguments.action)
if arguments.action == 'platform':
  os.system('uname -a')
elif arguments.action == 'setup':
  subprocess.run('deno i', cwd='platform/')
