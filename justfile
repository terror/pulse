set dotenv-load

default:
	just --list

alias f := fmt
alias r := run
alias t := test

all: build test clippy fmt-check

[group: 'misc']
build:
  cargo build

[group: 'check']
check:
 cargo check

[group: 'check']
ci: test clippy forbid
  cargo +nightly fmt --all -- --check
  cargo update --locked --package pulse

[group: 'check']
clippy:
  cargo clippy --all --all-targets

[group: 'format']
fmt:
  cargo +nightly fmt

[group: 'format']
fmt-check:
  cargo +nightly fmt --all -- --check

[group: 'check']
forbid:
  ./bin/forbid

[group: 'dev']
run *args:
  cargo run {{ args }}

[group: 'test']
test:
  cargo test

[group: 'dev']
watch +COMMAND='test':
  cargo watch --clear --exec "{{COMMAND}}"
