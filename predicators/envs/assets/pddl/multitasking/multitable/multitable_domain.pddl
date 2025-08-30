(define (domain multi_table_blocks)
  (:requirements :strips :typing)
  (:types
    block robot location - object
    table - location
  )

  (:predicates
    ;; blocksworld structure
    (on ?x - block ?y - block)
    (ontable ?x - block ?t - table)
    (clear ?x - block)
    (holding ?x - block)
    (handempty)

    ;; robot location graph
    (at ?r - robot ?l - location)
    (connected ?l1 - location ?l2 - location)

    ;; table-association for blocks ANYWHERE in the stack on that table
    (attable ?x - block ?t - table)
  )

  ;; Move the robot between locations
  (:action move
    :parameters (?r - robot ?from - location ?to - location)
    :precondition (and
      (at ?r ?from)
      (connected ?from ?to))
    :effect (and
      (at ?r ?to)
      (not (at ?r ?from)))
  )

  ;; Pick up a clear block that is sitting on a table
  (:action pickup
    :parameters (?r - robot ?x - block ?t - table)
    :precondition (and
      (handempty)
      (clear ?x)
      (ontable ?x ?t)
      (at ?r ?t))
    :effect (and
      (holding ?x)
      (not (handempty))
      (not (ontable ?x ?t))
      (not (attable ?x ?t)))
  )

  ;; Put a held block down on a table
  (:action putdown
    :parameters (?r - robot ?x - block ?t - table)
    :precondition (and
      (holding ?x)
      (at ?r ?t))
    :effect (and
      (ontable ?x ?t)
      (attable ?x ?t)
      (clear ?x)
      (handempty)
      (not (holding ?x)))
  )

  ;; Unstack a clear block from atop another block (must be at the same table)
  (:action unstack
    :parameters (?r - robot ?x - block ?y - block ?t - table)
    :precondition (and
      (handempty)
      (clear ?x)
      (on ?x ?y)
      (attable ?y ?t)   ;; ensures we act at the correct table
      (at ?r ?t))
    :effect (and
      (holding ?x)
      (clear ?y)
      (not (handempty))
      (not (on ?x ?y))
      (not (attable ?x ?t)))
  )

  ;; Stack a held block onto a clear block (inherits table of the support)
  (:action stack
    :parameters (?r - robot ?x - block ?y - block ?t - table)
    :precondition (and
      (holding ?x)
      (clear ?y)
      (attable ?y ?t)  ;; both must be at this table
      (at ?r ?t))
    :effect (and
      (on ?x ?y)
      (clear ?x)
      (handempty)
      (attable ?x ?t)
      (not (holding ?x))
      (not (clear ?y)))
  )
)
