test.py:
	pipenv run python -m unittest test_CPMM.py -v

test.rs:
	rustc --test cpmm.rs -o cpmm_test && ./cpmm_test

proof.rs:
	GLIBC_TUNABLES=glibc.rtld.optional_static_tls=10240 kani --fail-fast -Z concrete-playback --concrete-playback print cpmm.rs

