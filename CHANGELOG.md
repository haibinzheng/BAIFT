# Changelog

## Camera-ready repository preparation

- Restore the supplied author list and citation information.
- Replace anonymous-review presentation with public artifact documentation.
- Add an offline structural check without model downloads or API calls. Automated workflow installation remains pending because the current GitHub authorization does not include workflow uploads.
- Repair path-only Python compatibility files and add an explicit packed-to-flat example converter; preserve supplied examples.
- Document actual implementation scope, missing reproduction inputs, fallback behavior, legacy metric keys and configuration discrepancies.
- Remove tracked Python bytecode from the current tree; retain Git history.

This preparation does not add experiments, modify historical paper results, or certify full reproducibility.

## Public source documentation review

- Correct the package description to use the paper's BAIFT name.
- Update scoring usage documentation to the current script path and required configuration.
- Require a supplied model path when starting local serving, replacing a machine-specific default.
