(define (problem room_manipulation_task0)
  (:domain room_domain)
  (:objects
    table sofa shelf - furniture
    book cup - item
    left_gripper right_gripper - gripper
    loc0 loc1 loc2 - location
  )
  (:init
    ;; Furniture at locations
    (at table loc0)
    (at sofa loc1)
    (at shelf loc2)

    ;; Robot initial position
    (robot_at loc0)

    ;; Items initial placement
    (item_at book table)
    (item_at cup sofa)

    ;; Gripper status
    (free left_gripper)
    (free right_gripper)

    ;; Reachability (assume all locations are mutually reachable)
    (reachable loc0 loc1)
    (reachable loc1 loc0)
    (reachable loc0 loc2)
    (reachable loc2 loc0)
    (reachable loc1 loc2)
    (reachable loc2 loc1)
  )
  (:goal (and
    ;; Goal: Move the book from the table to the shelf.
    (item_at book shelf)
  ))
)
