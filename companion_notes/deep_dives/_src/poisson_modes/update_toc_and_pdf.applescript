-- Update the deep dive's table of contents in Word, save it, export the PDF, close it.
-- Safety: Word's "active document" is re-resolved at every use, so every action goes through the
-- document's NAME, and nothing is saved or closed unless the active document is the deep dive itself.
-- Word is sandboxed: it saves the .docx it opened, but writes the PDF only inside its own container.
-- usage: osascript update_toc_and_pdf.applescript /abs/path.docx /abs/container/path.pdf
on run argv
	set docPath to item 1 of argv
	set pdfPath to item 2 of argv
	set docName to do shell script "basename " & quoted form of docPath
	with timeout of 900 seconds
		tell application "Microsoft Word"
			set isOpen to false
			try
				set n to name of document docName
				set isOpen to true
			end try
			if not isOpen then open (POSIX file docPath)
			set ok to false
			repeat 40 times
				try
					if (name of active document) is docName then
						set ok to true
						exit repeat
					end if
				end try
				try
					activate object (document docName)
				end try
				delay 0.5
			end repeat
			if not ok then return "ABORTED: " & docName & " never became the active document; nothing saved or closed"
			set theDoc to document docName
			set msg to "document " & (name of theDoc)
			try
				update (table of contents 1 of theDoc)
				set msg to msg & "; toc updated"
			on error e1
				set msg to msg & "; toc NOT updated: " & e1
			end try
			save theDoc
			save as theDoc file name pdfPath file format format PDF
			if (name of active document) is docName then
				close active document
				set msg to msg & "; closed"
			else
				set msg to msg & "; NOT closed (another document became active)"
			end if
		end tell
	end timeout
	return msg
end run
