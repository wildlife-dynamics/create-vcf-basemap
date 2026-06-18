.PHONY: compile

WORKFLOW_DIR = ecoscope-workflows-create-vcf-basemap-workflow

compile:
	@SPEC_TMP=$$(mktemp /tmp/vcf-basemap-spec.XXXXXX.yaml) && \
	sed 's|path: "REPO_ROOT"|path: "$(CURDIR)"|' spec.yaml > $$SPEC_TMP && \
	wt-compiler compile \
	  --spec=$$SPEC_TMP \
	  --pkg-name-prefix=ecoscope-workflows \
	  --results-env-var=ECOSCOPE_WORKFLOWS_RESULTS \
	  --clobber --install; \
	EXIT=$$?; rm -f $$SPEC_TMP; exit $$EXIT
