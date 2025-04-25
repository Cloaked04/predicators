(define (domain pick-blocks)
  (:requirements :strips :typing :conditional-effects)
  ;;------------------------------------------------------------------
  ;; Types
  ;;------------------------------------------------------------------
  (:types
      thing               ; anything that can support another thing
      block surface - thing)

  ;;------------------------------------------------------------------
  ;; Predicates
  ;;------------------------------------------------------------------
  (:predicates
      (on ?b - block ?p - thing)      ; ?b is sitting on ?p
      (clear ?p - thing)              ; nothing on top of ?p
      (holding ?b - block)            ; gripper is grasping ?b
      (handempty))                    ; gripper is free

  ;;------------------------------------------------------------------
  ;; Pick (can be “unstack” if ?p is a block, or table/tray grasp)
  ;;------------------------------------------------------------------
  (:action pick
    :parameters (?b - block ?p - thing)
    :precondition (and (on ?b ?p) (clear ?b) (handempty))
    :effect (and
        ;; add effects
        (holding ?b)
        ;; delete effects
        (not (on ?b ?p))
        (not (handempty))
        (not (clear ?b))
        ;; conditional: the support becomes clear after the pick
        (when (on ?b ?p) (clear ?p))))

  ;;------------------------------------------------------------------
  ;; Place  (stack on a block or put on a surface)
  ;;------------------------------------------------------------------
  (:action place
    :parameters (?b - block ?p - thing)
    :precondition (and (holding ?b) (clear ?p))
    :effect (and
        ;; add effects
        (on ?b ?p)
        (handempty)
        (clear ?b)
        ;; delete effects
        (not (holding ?b))
        (not (clear ?p))))
)
