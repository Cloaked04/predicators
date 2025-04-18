(define (problem room_manipulation_task_multitask)
  (:domain room_domain)
  (:objects
    table sofa shelf desk - furniture
    book cup remote keys phone pen magazine notebook - item
    left_gripper right_gripper - gripper
    loc0 loc1 loc2 loc3 loc4 - location
  )
  (:init
    ;; Place furniture at specific locations.
    (at table loc0)
    (at sofa loc1)
    (at shelf loc2)
    (at desk loc3)

    ;; Set the robot's initial position.
    (robot_at loc4)

    ;; Place items initially.
    (item_at book table)
    (item_at cup table)
    (item_at remote sofa)
    (item_at keys desk)
    (item_at phone shelf)
    (item_at pen desk)
    (item_at magazine sofa)
    (item_at notebook table)

    ;; All grippers start off free.
    (free left_gripper)
    (free right_gripper)

    ;; Assume all locations are mutually reachable.
    (reachable loc0 loc1) (reachable loc1 loc0)
    (reachable loc0 loc2) (reachable loc2 loc0)
    (reachable loc0 loc3) (reachable loc3 loc0)
    (reachable loc0 loc4) (reachable loc4 loc0)
    (reachable loc1 loc2) (reachable loc2 loc1)
    (reachable loc1 loc3) (reachable loc3 loc1)
    (reachable loc1 loc4) (reachable loc4 loc1)
    (reachable loc2 loc3) (reachable loc3 loc2)
    (reachable loc2 loc4) (reachable loc4 loc2)
    (reachable loc3 loc4) (reachable loc4 loc3)
  )
  (:goal (and
    ;; - Move the book from the table to the shelf.
    (item_at book shelf)
    ;; - Move the cup from the table to the sofa.
    (item_at cup sofa)
    ;; - Move the remote from the sofa to the table.
    (item_at remote table)
    ;; - Move the keys from the desk to the shelf.
    (item_at keys shelf)
    ;; - Move the phone from the shelf to the desk.
    (item_at phone desk)
    ;; - Move the pen from the desk to the sofa.
    (item_at pen sofa)
    ;; - Move the magazine from the sofa to the table.
    (item_at magazine table)
    ;; - Move the notebook from the table to the desk.
    (item_at notebook desk)
  ))
)
