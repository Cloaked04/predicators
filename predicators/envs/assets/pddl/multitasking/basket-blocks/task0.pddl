(define (problem move-blocks-no-limit)
  (:domain basket-moving-no-limit)
  
  (:objects
    table1 table2 table3 - table
    block1 block2 block3 block4 - block
    basket1 - basket
  )
  
  (:init
    ;; All four blocks are initially on table1.
    (at block1 table1)
    (at block2 table1)
    (at block3 table1)
    (at block4 table1)
    
    ;; The basket is initially on table3.
    (basket-at table3)
  )
  
  (:goal 
    (and
      ;; All blocks must be moved to table2.
      (at block1 table2)
      (at block2 table2)
      (at block3 table2)
      (at block4 table2)
    )
  )
)
