.PHONY: all assets validate clean

all:
	./build.sh

assets:
	@if command -v uv >/dev/null 2>&1; then uv run python prepare_assets.py; else python3 prepare_assets.py; fi

validate:
	python3 validate_submission.py

clean:
	rm -rf build
	rm -f Team_QTyche_Phase3.pdf
