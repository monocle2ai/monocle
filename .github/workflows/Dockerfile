FROM node:22-alpine

WORKDIR /repolinter/repo

RUN apk add --no-cache \
  git \
  && rm -rf /var/cache/apk/*

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]