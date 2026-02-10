# Makefile for ICML 2026 Paper Compilation
# Insurance-GSSM: Domain-Adapted State-Space Models

# Variables
MAIN = icml2026_insurance_gssm
SUPP = supplementary_material
LATEX = pdflatex
BIBTEX = bibtex
VIEWER = open  # Use 'evince' on Linux, 'xdg-open' on some systems

# Directories
FIGDIR = figures
COMPILED = compiled

# Targets
.PHONY: all main supp clean cleanall view viewsupp help setup

# Default target: build everything
all: setup main supp

# Build main paper
main: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib icml2026.sty
	@echo "========================================="
	@echo "Compiling Main Paper..."
	@echo "========================================="
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)
	@mkdir -p $(COMPILED)
	@mv $(MAIN).pdf $(COMPILED)/
	@echo "✓ Main paper compiled: $(COMPILED)/$(MAIN).pdf"

# Build supplementary material
supp: $(SUPP).pdf

$(SUPP).pdf: $(SUPP).tex references.bib
	@echo "========================================="
	@echo "Compiling Supplementary Material..."
	@echo "========================================="
	$(LATEX) $(SUPP)
	$(BIBTEX) $(SUPP)
	$(LATEX) $(SUPP)
	$(LATEX) $(SUPP)
	@mkdir -p $(COMPILED)
	@mv $(SUPP).pdf $(COMPILED)/
	@echo "✓ Supplementary material compiled: $(COMPILED)/$(SUPP).pdf"

# Setup: create directories and copy figures
setup:
	@echo "Setting up paper directory..."
	@mkdir -p $(FIGDIR)
	@mkdir -p $(COMPILED)
	@if [ -d "../results/figures/paper" ]; then \
		echo "Copying paper figures..."; \
		cp ../results/figures/paper/*.pdf $(FIGDIR)/ 2>/dev/null || echo "No paper figures found"; \
	fi
	@if [ -d "../results/figures/enhanced" ]; then \
		echo "Copying enhanced figures..."; \
		cp ../results/figures/enhanced/*.pdf $(FIGDIR)/ 2>/dev/null || echo "No enhanced figures found"; \
	fi
	@echo "✓ Setup complete"

# View compiled PDFs
view: $(COMPILED)/$(MAIN).pdf
	@echo "Opening main paper..."
	$(VIEWER) $(COMPILED)/$(MAIN).pdf

viewsupp: $(COMPILED)/$(SUPP).pdf
	@echo "Opening supplementary material..."
	$(VIEWER) $(COMPILED)/$(SUPP).pdf

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	@rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot
	@rm -f *.fdb_latexmk *.fls *.synctex.gz
	@echo "✓ Cleaned"

# Clean everything including PDFs
cleanall: clean
	@echo "Removing compiled PDFs..."
	@rm -rf $(COMPILED)
	@echo "✓ All clean"

# Quick recompile (no bibtex)
quick:
	@echo "Quick recompile (no bibliography update)..."
	$(LATEX) $(MAIN)
	@mkdir -p $(COMPILED)
	@mv $(MAIN).pdf $(COMPILED)/
	@echo "✓ Quick compile done"

# Count words (approximate)
wordcount:
	@echo "Approximate word count:"
	@echo "Main paper:"
	@detex $(MAIN).tex | wc -w
	@echo "Supplementary:"
	@detex $(SUPP).tex | wc -w

# Check for common LaTeX errors
check:
	@echo "Checking for common issues..."
	@echo "- Missing references:"
	@grep -n "??" $(MAIN).log 2>/dev/null || echo "  None found"
	@echo "- Overfull boxes:"
	@grep -n "Overfull" $(MAIN).log 2>/dev/null | head -5 || echo "  None found"
	@echo "- Undefined references:"
	@grep -n "undefined" $(MAIN).log 2>/dev/null || echo "  None found"

# Package for submission
package: all
	@echo "Creating submission package..."
	@mkdir -p submission
	@cp $(COMPILED)/$(MAIN).pdf submission/
	@cp $(COMPILED)/$(SUPP).pdf submission/
	@cp -r $(FIGDIR) submission/
	@cp *.tex *.bib *.sty submission/
	@tar -czf submission.tar.gz submission/
	@echo "✓ Package created: submission.tar.gz"

# Update figures from experiments
update-figures:
	@echo "Updating figures from experiment results..."
	@cd ../experiments && python3 enhanced_visualizations.py
	@cd ../experiments && python3 additional_visualizations.py
	@$(MAKE) setup
	@echo "✓ Figures updated"

# Help
help:
	@echo "========================================="
	@echo "Insurance-GSSM Paper Makefile"
	@echo "========================================="
	@echo ""
	@echo "Targets:"
	@echo "  make all          - Compile main paper and supplementary"
	@echo "  make main         - Compile main paper only"
	@echo "  make supp         - Compile supplementary only"
	@echo "  make quick        - Quick recompile (no bibtex)"
	@echo "  make view         - Open main paper PDF"
	@echo "  make viewsupp     - Open supplementary PDF"
	@echo "  make clean        - Remove auxiliary files"
	@echo "  make cleanall     - Remove all generated files"
	@echo "  make setup        - Copy figures from experiments"
	@echo "  make update-figures - Regenerate experiment figures"
	@echo "  make wordcount    - Count words in documents"
	@echo "  make check        - Check for LaTeX errors"
	@echo "  make package      - Create submission package"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  1. First run: make all"
	@echo "  2. After changes: make quick (faster)"
	@echo "  3. After experiments: make update-figures && make all"
	@echo ""
	@echo "========================================="
