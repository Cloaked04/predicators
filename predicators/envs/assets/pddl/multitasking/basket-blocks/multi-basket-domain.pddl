(define (domain basket-moving-multi)
  (:requirements :typing :strips)
  (:types block table basket)
  
  (:predicates 
    (at ?blk - block ?t - table)           ; A block is on a table.
    (basket-at ?bkt - basket ?t - table)     ; A basket is at a given table.
    (in-basket ?blk - block ?bkt - basket)   ; A block is inside a basket.
  )

  ;; Action: Move a basket from one table to another.
  (:action move-basket
    :parameters (?bkt - basket ?from - table ?to - table)
    :precondition (basket-at ?bkt ?from)
    :effect (and 
              (not (basket-at ?bkt ?from))
              (basket-at ?bkt ?to)
            )
  )
  
  ;; Action: Load a block from a table into a basket.
  (:action load-block
    :parameters (?blk - block ?bkt - basket ?t - table)
    :precondition (and 
                    (at ?blk ?t)
                    (basket-at ?bkt ?t)
                  )
    :effect (and 
              (in-basket ?blk ?bkt)
              (not (at ?blk ?t))
            )
  )
  
  ;; Action: Unload a block from a basket onto a table.
  (:action unload-block
    :parameters (?blk - block ?bkt - basket ?t - table)
    :precondition (and 
                    (in-basket ?blk ?bkt)
                    (basket-at ?bkt ?t)
                  )
    :effect (and 
              (at ?blk ?t)
              (not (in-basket ?blk ?bkt))
            )
  )
)
