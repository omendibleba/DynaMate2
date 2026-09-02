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
        className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-green-300 bg-green-50/50 px-4 py-6 text-center transition hover:bg-green-50 dark:border-green-800 dark:bg-green-950/20"
      >
        <span className="text-sm font-medium text-green-700 dark:text-green-400">
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
        <p className="mt-2 rounded-lg bg-green-50 px-3 py-2 font-mono text-xs text-green-700 dark:bg-green-950/40 dark:text-green-400">
          Saved path (prompt auto-filled): {savedPath}
        </p>
      )}
      {error && (
        <p className="mt-2 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700 dark:bg-red-950/40 dark:text-red-400">
          ⚠ {error}
        </p>
      )}
    </div>
  )
}
