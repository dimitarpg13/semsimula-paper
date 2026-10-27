#!/bin/sh
# Update the TOC in Word, save the .docx, export the PDF via Word's container, copy it next to the .docx.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); DD=$(cd "$HERE/../.." && pwd)
NAME=Deep_Dive_in_the_Poisson_Mode_Registers_Wells_and_Semantic_Templates
CONT="$HOME/Library/Containers/com.microsoft.Word/Data/Documents"
rm -f "$CONT/$NAME.pdf"
osascript "$HERE/update_toc_and_pdf.applescript" "$DD/$NAME.docx" "$CONT/$NAME.pdf"
cp "$CONT/$NAME.pdf" "$DD/$NAME.pdf" && echo "wrote $DD/$NAME.pdf"
