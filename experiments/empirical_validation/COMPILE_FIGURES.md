# How to Compile FIGURES_DOCUMENT.tex

## Quick Compilation

### Option 1: Command Line (if you have pdflatex)
```bash
cd /Users/nhassen/Documents/AIML/Insurance/Insurance-Forecasting-GSSM/experiments/empirical_validation

pdflatex FIGURES_DOCUMENT.tex
pdflatex FIGURES_DOCUMENT.tex  # Run twice for references
```

### Option 2: Overleaf (Recommended)
1. Go to https://www.overleaf.com/
2. Create new project → Upload Project
3. Upload these files:
   - `FIGURES_DOCUMENT.tex`
   - `figures/figure1_autocorrelation_analysis.pdf`
   - `figures/figure2_cycle_detection_fft.pdf`
   - `figures/figure3_flow_selectivity.pdf`
4. Click "Recompile"

### Option 3: Local LaTeX Editor
- **TeXShop** (Mac): Open FIGURES_DOCUMENT.tex → Click "Typeset"
- **TeXstudio** (Windows/Mac/Linux): Open → F5 to compile
- **VSCode + LaTeX Workshop**: Open → Build (Ctrl+Alt+B)

## What You Get

A comprehensive PDF document (~10-12 pages) containing:

### Structure
1. **Title & Abstract** - Overview of empirical validation
2. **Introduction** - Methodology and approach
3. **RQ1: Autocorrelation** (2 pages)
   - Theoretical foundation (Definition 3.4)
   - Full-page Figure 1 with 6 panels
   - Detailed caption linking theory to results
   - Key findings summary
4. **RQ2: Cycle Detection** (2 pages)
   - Theoretical foundation (Theorem 3.2)
   - Full-page Figure 2 with 6 panels
   - FFT analysis and market phases
   - Key findings summary
5. **RQ3: Flow-Selectivity** (2 pages)
   - Theoretical foundation (Proposition 3.1)
   - Full-page Figure 3 with 6 panels
   - Gating mechanism analysis
   - Key findings summary
6. **Statistical Summary** - T-tests, p-values, Cohen's d
7. **Theory-Practice Mapping** - Explicit links
8. **Conclusion** - Key contributions
9. **Bibliography** - All references
10. **Appendix** - Reproducibility & specifications

### Features
- ✅ Publication-quality formatting
- ✅ All figures at 300 DPI
- ✅ Detailed 150-250 word captions per figure
- ✅ Statistical validation tables
- ✅ Theory-to-experiment mapping
- ✅ Complete citations
- ✅ Reproducibility instructions

## Expected Output

**PDF Properties**:
- Pages: ~10-12
- Size: ~2-3 MB (with vector graphics)
- Format: Two-column layout (standard conference format)
- Quality: Publication-ready for ICML 2026

**Content Quality**:
- Each figure panel explained in caption
- Theoretical concepts (Definitions, Theorems, Propositions) stated formally
- Empirical results linked to theory explicitly
- Statistical significance reported for all claims
- Key findings highlighted with improvement percentages

## Troubleshooting

### Missing Figures Error
If you get "File not found" errors:
```bash
# Make sure figures are in the right location
ls figures/*.pdf

# Should show:
# figures/figure1_autocorrelation_analysis.pdf
# figures/figure2_cycle_detection_fft.pdf
# figures/figure3_flow_selectivity.pdf
```

### Package Errors
If missing LaTeX packages:
```bash
# On Mac with MacTeX:
sudo tlmgr install <package-name>

# On Linux with TeX Live:
sudo apt-get install texlive-full
```

### Font Issues
The document uses standard serif fonts. If you prefer:
```latex
% Change in preamble:
\usepackage{times}  % For Times font
\usepackage{helvet} % For Helvetica
```

## Integration with Main Paper

To include these figures in your main paper:

```latex
% In your main ICML paper tex file:

% Section 5: Experiments
\section{Experimental Validation}

% RQ1
\subsection{Autocorrelation Analysis}
As shown in Figure~\ref{fig:autocorrelation}, the autocorrelation 
module achieves...

\input{experiments/empirical_validation/figure_section_rq1.tex}

% Or simply:
\begin{figure*}[t]
    \includegraphics[width=\linewidth]{experiments/empirical_validation/figures/figure1_autocorrelation_analysis.pdf}
    \caption{See FIGURES_DOCUMENT.tex for full caption}
    \label{fig:autocorrelation}
\end{figure*}
```

## Files Included

The document references:
- `FIGURES_DOCUMENT.tex` - Main LaTeX source (275 lines)
- `figures/figure1_autocorrelation_analysis.pdf` - RQ1 (6 panels)
- `figures/figure2_cycle_detection_fft.pdf` - RQ2 (6 panels)
- `figures/figure3_flow_selectivity.pdf` - RQ3 (6 panels)

All files committed to git and available at:
`experiments/empirical_validation/`

## Next Steps

1. ✅ **Compile the document** to see the full output
2. ✅ **Review figure quality** - Check if panels are clear
3. ✅ **Read captions** - Verify theory-practice links are accurate
4. ✅ **Check statistics** - Ensure p-values and effect sizes are correct
5. ✅ **Integrate into paper** - Copy relevant sections to main manuscript

## Quick Preview

You can also view figures individually:
```bash
# Open each PDF directly
open figures/figure1_autocorrelation_analysis.pdf
open figures/figure2_cycle_detection_fft.pdf
open figures/figure3_flow_selectivity.pdf
```

Or use Preview/Acrobat to view them side-by-side before compiling.

---

**Created**: February 7, 2026  
**Author**: Nadhir Hassen  
**Location**: `experiments/empirical_validation/FIGURES_DOCUMENT.tex`  
**Status**: ✅ Ready to compile
