#! /usr/bin/env python
#
# Copyright (c) 2014 Kees Bakker.  All rights reserved.
#
# This file is part of Mitsuta.
#
# Mitsuta is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published bythe Free Software Foundation, either version 3 of
# the License, or(at your option) any later version.
#
# Mitsuta is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mitsuta.  If not, see
# <http://www.gnu.org/licenses/>.
#
# This Python script is meant to analyse the Mitsuta algorithm
# with some test vectors.
#
# Details about the Mitsuta algorithm can be found at:
#    http://abelian.org/vlf/bearings.html
#
# There are two alternatives (mitsuta_mean2 and mitsuta_mean3)
# which avoid the if-else statements. They can be cheaper in
# code size (for example on an Atmel processor).

import sys
import random

def mitsuta_mean(b, branch=360):
    half_branch = branch / 2
    D = b[0]
    mysum = D
    for val in b[1:]:
        delta = val - D
        if delta < -half_branch:
            D = D + delta + branch
        elif delta < half_branch:
            D = D + delta
        else:
            D = D + delta - branch
        mysum = mysum + D
    m = mysum / len(b)
    return (m + branch) % branch

def mitsuta_mean2(b, branch=360):
    half_branch = branch / 2
    D = b[0]
    mysum = D
    for val in b[1:]:
        delta = val - (D - half_branch)
        D = (D - half_branch) + (delta % branch)
        mysum = mysum + D
    m = mysum / len(b)
    return (m + branch) % branch

def mitsuta_mean3(b, doDiag=False, branch=360):
    half_branch = branch / 2
    D = b[0]
    mysum = D
    n = 1
    for val in b[1:]:
        if doDiag:
            print("_d %d" % D)
        D = D - half_branch
        delta = val - D
        if doDiag:
            print("delta %d" % delta)
            print("delta %% 360 %d" % (delta % branch))
        D = D + (delta % branch)
        mysum = mysum + D
        n = n + 1
        if doDiag:
            m = mysum / n
            print("mean %d" % ((m + branch) % branch))
    m = mysum / len(b)
    return (m + branch) % branch

def doTest(vec):
    nm, vec = vec
    m = mitsuta_mean(vec)
    m2 = mitsuta_mean2(vec)
    if m != m2:
        print("m=%d <> m2=%d" % (m, m2))
    m3 = mitsuta_mean3(vec, True)
    if m != m3:
        print("m=%d <> m3=%d" % (m, m3))
    print("%s: %s => %d" % (nm, vec, m))

if __name__ == '__main__':

    # unstable wind direction tests
    testvec7 = [345,342,342,342,22,67,12,330,0,36]
    doTest(["testvec7", testvec7])
    sys.exit(0)

    testvec1 = [350, 351, 352, 2, 3, 4]
    doTest(["testvec1", testvec1])

    testvec2 = [(v - 180) % 360 for v in testvec1]
    doTest(["testvec2", testvec2])

    testvec3 = [300, 351, 310, 290, 299, 200]
    doTest(["testvec3", testvec3])

    for i in range(20):
        for j in range(20):
            vec = [359 - j, i]
            doTest(["testvec4", vec])

    for i in range(20):
        for j in range(20):
            vec = [i, 359 - j]
            doTest(["testvec5", vec])

    for i in range(20):
        testvec6 = [j for j in range(i)] + [350]
        doTest(["testvec6", testvec6])
        random.shuffle(testvec6)
        doTest(["testvec6", testvec6])
        random.shuffle(testvec6)
        doTest(["testvec6", testvec6])
        random.shuffle(testvec6)
        doTest(["testvec6", testvec6])
