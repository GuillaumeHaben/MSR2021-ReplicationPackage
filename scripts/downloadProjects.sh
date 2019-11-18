#!/bin/sh

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 gitURL" >&2
  exit 1
fi

ADDRESS=$1
git clone $ADDRESS