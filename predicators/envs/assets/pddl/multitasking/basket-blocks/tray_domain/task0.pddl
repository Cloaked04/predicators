(define (problem buried-blue)
  (:domain pick-blocks)
  ;;---------------------------------------------------------------
  ;; Objects
  ;;---------------------------------------------------------------
  (:objects
      red green yellow orange blue       - block
      table-a table-b tray               - surface)

  ;;---------------------------------------------------------------
  ;; Initial state  (blue is at bottom of a 5-block stack on table-a)
  ;;---------------------------------------------------------------
  (:init
      ;; stack from bottom → top
      (on blue   table-a)
      (on orange blue)
      (on yellow orange)
      (on green  yellow)
      (on red    green)

      (clear red)        ; top of the stack
      (clear table-b)    ; destination table is empty
      (clear tray)       ; tray starts empty

      (handempty))

  ;;---------------------------------------------------------------
  ;; Goal
  ;;---------------------------------------------------------------
  (:goal
      (and (on blue table-b)))
)
