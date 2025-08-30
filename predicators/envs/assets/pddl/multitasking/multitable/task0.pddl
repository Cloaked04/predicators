(define (problem two_tables_bottom_to_other_table)
  (:domain multi_table_blocks)

  (:objects
    robby - robot
    home - location
    table1 table2 - table
    A B C - block
  )

  (:init
    ;; graph connectivity (make it symmetric for simplicity)
    (connected home table1)
    (connected table1 home)
    (connected home table2)
    (connected table2 home)

    ;; robot initially at home
    (at robby home)

    ;; initial 3-high stack on table1: A on B on C on table1
    (on A B)
    (on B C)
    (ontable C table1)

    ;; table-association for the whole stack at table1
    (attable A table1)
    (attable B table1)
    (attable C table1)

    ;; block states
    (clear A)
    (handempty)
  )

  ;; Goal: bottom block C ends up on table2.
  ;; (The planner must unstack A and B to free C.)
  (:goal
    (and (ontable C table2))
  )
)
