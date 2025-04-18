(define (problem move-red-blocks-multi)
  (:domain basket-moving-multi)
  
  (:objects
    table1 table2 table3 table4 - table
    basket1 basket2 - basket
    red1 red2 red3 - block
    blue1 blue2 - block
  )
  
  (:init
    ;; Red blocks are on table1.
    (at red1 table1)
    (at red2 table1)
    (at red3 table1)
    
    ;; Blue blocks are already in basket1.
    (in-basket blue1 basket1)
    (in-basket blue2 basket1)
    
    ;; Basket locations:
    (basket-at basket1 table3)  ; basket1 (with blue blocks) is at table3.
    (basket-at basket2 table2)  ; basket2 is at table2 and is empty.
  )
  
  (:goal (and
    ;; All red blocks should be on table4.
    (at red1 table4)
    (at red2 table4)
    (at red3 table4)
  ))
)
