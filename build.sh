#!/usr/bin/env bash

function docmd
{
echo -n "$*..."
if $* ; then
	echo "ok"
else
	echo "failed"
	exit 1
fi
}

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f mccl/config/config.h ] || [ "x$1" != "x" ]; then
docmd autoreconf --install
docmd ./configure
fi
docmd make all
docmd make check
