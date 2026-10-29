"use client";

import { useRef, useState } from "react";
import { CheckCircle2, FileImage, UploadCloud, X } from "lucide-react";

type Props = {
  file: File | null;
  onFile: (file: File | null) => void;
  accept?: string;
  helperText?: string;
};

export function UploadDropzone({ file, onFile, accept, helperText }: Props) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function pick(files: FileList | null) {
    if (files?.length) onFile(files[0]);
  }

  return (
    <div
      className={`dropzone ${dragging ? "is-dragging" : ""}`}
      onDragEnter={(e) => { e.preventDefault(); setDragging(true); }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={(e) => { e.preventDefault(); setDragging(false); }}
      onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files); }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") inputRef.current?.click(); }}
    >
      <UploadCloud size={30} color="var(--green)" />
      <div className="dropzone-title">{dragging ? "Release to upload" : "Drop a goat image here"}</div>
      <div className="dropzone-copy">{helperText || "or click to browse · JPG, PNG, BMP or WEBP"}</div>

      {file && (
        <div className="dropzone-file" onClick={(e) => e.stopPropagation()}>
          <FileImage size={14} color="var(--green)" />
          <span style={{maxWidth:310,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{file.name}</span>
          <CheckCircle2 size={14} color="var(--green)" />
          <button
            aria-label="Remove file"
            style={{border:0,background:"transparent",padding:0,display:"grid"}}
            onClick={() => onFile(null)}
          >
            <X size={14} color="var(--muted)" />
          </button>
        </div>
      )}

      <input
        ref={inputRef}
        hidden
        type="file"
        accept={accept || ".jpg,.jpeg,.png,.bmp,.webp,image/jpeg,image/png,image/bmp,image/webp"}
        onChange={(e) => pick(e.target.files)}
      />
    </div>
  );
}
