import { useEffect, useRef } from 'react'

interface DrawingCanvasProps {
  size: number
  strokeWidth: number
  onStrokeEnd: (imageDataUrl: string) => void
  onClear: () => void
}

interface Point {
  x: number
  y: number
}

function pointFromEvent(event: React.PointerEvent<HTMLCanvasElement>): Point {
  const canvas = event.currentTarget
  const rect = canvas.getBoundingClientRect()
  return {
    x: ((event.clientX - rect.left) * canvas.width) / rect.width,
    y: ((event.clientY - rect.top) * canvas.height) / rect.height,
  }
}

export function DrawingCanvas({ size, strokeWidth, onStrokeEnd, onClear }: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const isDrawingRef = useRef(false)
  const lastPointRef = useRef<Point>({ x: 0, y: 0 })

  useEffect(() => {
    const context = canvasRef.current?.getContext('2d')
    if (!context) return
    context.fillStyle = '#ffffff'
    context.fillRect(0, 0, size, size)
  }, [size])

  function handlePointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    if (event.button !== 0) return
    const context = canvasRef.current?.getContext('2d')
    if (!context) return

    isDrawingRef.current = true
    lastPointRef.current = pointFromEvent(event)
    context.fillStyle = '#000000'
    context.beginPath()
    context.arc(lastPointRef.current.x, lastPointRef.current.y, strokeWidth / 2, 0, Math.PI * 2)
    context.fill()
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  function handlePointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    // Require the primary button to still be held, not just our own `isDrawingRef` —
    // guards against a stray pointermove (hover, scroll) silently extending a stroke.
    if (!isDrawingRef.current || !(event.buttons & 1)) return
    const context = canvasRef.current?.getContext('2d')
    if (!context) return

    const point = pointFromEvent(event)
    context.strokeStyle = '#000000'
    context.lineWidth = strokeWidth
    context.lineCap = 'round'
    context.lineJoin = 'round'
    context.beginPath()
    context.moveTo(lastPointRef.current.x, lastPointRef.current.y)
    context.lineTo(point.x, point.y)
    context.stroke()
    lastPointRef.current = point
  }

  function endStroke(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!isDrawingRef.current) return
    isDrawingRef.current = false
    event.currentTarget.releasePointerCapture(event.pointerId)
    onStrokeEnd(event.currentTarget.toDataURL('image/png'))
  }

  function handleClear() {
    const context = canvasRef.current?.getContext('2d')
    if (!context) return
    context.fillStyle = '#ffffff'
    context.fillRect(0, 0, size, size)
    onClear()
  }

  return (
    <div className="drawing-canvas">
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        className="drawing-canvas__surface"
        aria-label="Drawing surface. Draw one handwritten digit with a pointer or touch."
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endStroke}
        onPointerLeave={endStroke}
        onPointerCancel={endStroke}
      />
      <button type="button" className="drawing-canvas__clear" onClick={handleClear}>
        Clear drawing
      </button>
    </div>
  )
}
