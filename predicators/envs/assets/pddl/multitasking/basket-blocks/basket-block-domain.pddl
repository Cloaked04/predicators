(define (domain basket-moving-no-limit)
  (:requirements :typing :strips)
  (:types block table basket)
  
  (:predicates 
    (at ?b - block ?t - table)          ; block b is on table t
    (basket-at ?t - table)              ; basket is at table t
    (in-basket ?b - block)              ; block b is inside the basket
  )
  
  ;; Action: Move the basket from one table to another.
  (:action move-basket
    :parameters (?from ?to - table)
    :precondition (basket-at ?from)
    :effect (and 
              (not (basket-at ?from))
              (basket-at ?to)
            )
  )
  
  ;; Action: Load a block into the basket from the table where the basket is.
  (:action load-block
    :parameters (?b - block ?t - table)
    :precondition (and 
                    (at ?b ?t)
                    (basket-at ?t)
                  )
    :effect (and 
              (not (at ?b ?t))
              (in-basket ?b)
            )
  )
  
  ;; Action: Unload a block from the basket onto the table where the basket is.
  (:action unload-block
    :parameters (?b - block ?t - table)
    :precondition (and 
                    (in-basket ?b)
                    (basket-at ?t)
                  )
    :effect (and 
              (at ?b ?t)
              (not (in-basket ?b))
            )
  )
)
