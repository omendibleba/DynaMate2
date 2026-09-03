import { AlertTriangle, Upload } from 'lucide-react'
import { useRef, useState } from 'react'
import { uploadTool } from '../lib/api'

interface FileUploadZoneProps {
  onUploaded: (prompt: string) => void
}

export function FileUploadZone({ onUploaded }: FileUploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [savedPath, setSavedPath] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)

  async function handleFile(file: File) {
    setError(null)
    setIsUploading(true)
    try {
      const { path, prompt } = await uploadTool(file)
      setSavedPath(path)
      onUploaded(prompt)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div>
      <label
        htmlFor="tool-upload"
        className="flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-xl border-2 border-dashed border-emerald-300 bg-emerald-50/50 px-4 py-6 text-center transition hover:bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950/20"
      >
        <Upload className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
        <span className="text-sm font-medium text-emerald-700 dark:text-emerald-400">
          {isUploading ? 'Uploading…' : 'Upload a Python tool script (.py)'}
        </span>
        <input
          id="tool-upload"
          ref={inputRef}
          type="file"
          accept=".py"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) handleFile(file)
            e.target.value = ''
          }}
        />
      </label>

      {savedPath && (
        <p className="mt-2 rounded-xl bg-emerald-50 px-3 py-2 font-mono text-xs text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400">
          Saved path (prompt auto-filled): {savedPath}
        </p>
      )}
      {error && (
        <p className="mt-2 flex items-center gap-1.5 rounded-xl bg-red-50 px-3 py-2 text-xs text-red-700 dark:bg-red-950/40 dark:text-red-400">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" /> {error}
        </p>
      )}
    </div>
  )
}
