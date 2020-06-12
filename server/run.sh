#!/usr/bin/env sh

uwsgi --http :5088 --plugins python37 --master --mount /=app:app
