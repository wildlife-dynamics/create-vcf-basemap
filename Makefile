.PHONY: compile

WORKFLOW_DIR = ecoscope-workflows-create-vcf-basemap-workflow

compile:
	wt-compiler compile \
	  --spec=spec.yaml \
	  --pkg-name-prefix=ecoscope-workflows \
	  --results-env-var=ECOSCOPE_WORKFLOWS_RESULTS \
	  --clobber --install
