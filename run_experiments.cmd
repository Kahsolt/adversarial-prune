@ECHO OFF
SETLOCAL

SET MODEL=resnet18
SET PYTHON=python.exe

%PYTHON% prune.py -M %MODEL% -E 1e-5
%PYTHON% prune.py -M %MODEL% -E 1e-4
%PYTHON% prune.py -M %MODEL% -E 1e-3
%PYTHON% prune.py -M %MODEL% -E 5e-3
%PYTHON% prune.py -M %MODEL% -E 1e-2
%PYTHON% prune.py -M %MODEL% -E 5e-2
%PYTHON% prune.py -M %MODEL% -E 1e-1

%PYTHON% test.py --ckpt out\%MODEL%_1e-5.pth
%PYTHON% test.py --ckpt out\%MODEL%_1e-4.pth
%PYTHON% test.py --ckpt out\%MODEL%_1e-3.pth
%PYTHON% test.py --ckpt out\%MODEL%_5e-3.pth
%PYTHON% test.py --ckpt out\%MODEL%_1e-2.pth
%PYTHON% test.py --ckpt out\%MODEL%_5e-2.pth
%PYTHON% test.py --ckpt out\%MODEL%_1e-1.pth
