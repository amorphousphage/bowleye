class Frame:
    """
    Represents a single frame in a bowling game.
    Stores the pins knocked down in each roll.
    """
    def __init__(self, index):
        self.index = index
        self.rolls = []          # e.g. [7, 2]
        self.fallen_pins = []    # list of sets: [{1,2,4}, {7}]
        self.score = None        # cumulative score (computed later)

    def AddShot(self, fallen_pins):
        """
        Store the fallen pins for this shot.
        """
        self.fallen_pins.append(set(fallen_pins))
        self.rolls.append(len(fallen_pins))

    def IsStrike(self):
        return len(self.rolls) >= 1 and self.rolls[0] == 10

    def IsSpare(self):
        return len(self.rolls) >= 2 and sum(self.rolls[:2]) == 10 and not self.is_strike()

    def IsComplete(self):
        """
        Determines whether the frame is complete.
        """
        # Frames 1–9
        if self.index < 10:
            # Strike ends the frame
            if self.IsStrike():
                return True
            # Two rolls ends the frame
            if len(self.rolls) == 2:
                return True
            return False

        # Frame 10
        if len(self.rolls) == 3:
            return True
        if len(self.rolls) == 2:
            # No bonus roll unless strike or spare
            if sum(self.rolls[:2]) < 10:
                return True
        return False


class GameManager:
    """
    Manages the entire game state:
    - frames
    - current frame/ball
    - scoring logic
    """
    def __init__(self):
        self.ResetGame()

    def ResetGame(self):
        self.frames = [Frame(i+1) for i in range(10)]
        self.current_frame_index = 0   # 0-based index
        self.game_over = False

    @property
    def CurrentFrame(self):
        return self.frames[self.current_frame_index]

    def ProcessShot(self, fallen_pins):
        """
        Called when the coordinator sends fallen pins.
        This updates the game state and advances the frame/ball.
        """
        if self.game_over:
            print("Game is already complete.")
            return

        frame = self.current_frame
        frame.AddShot(fallen_pins)

        # Advance frame if needed
        if frame.IsComplete():
            if self.current_frame_index == 9:
                # 10th frame complete
                self.game_over = True
            else:
                self.current_frame_index += 1

        # Recalculate scores after every shot
        self.RecalculateScores()

    def RecalculateScores(self):
        """
        Full bowling scoring logic:
        - Strike: 10 + next two rolls
        - Spare: 10 + next one roll
        - Open frame: sum of rolls
        - 10th frame: sum of all rolls
        """
        running_total = 0

        # Flatten all rolls for easy bonus lookup
        all_rolls = []
        for f in self.frames:
            for r in f.rolls:
                all_rolls.append(r)

        # Build a list of (frame_index, roll_index_in_flat_list)
        roll_pointer = []
        flat_index = 0
        for f in self.frames:
            indices = []
            for _ in f.rolls:
                indices.append(flat_index)
                flat_index += 1
            roll_pointer.append(indices)

        # Compute scores frame by frame
        for i, frame in enumerate(self.frames):
            if not frame.rolls:
                frame.score = None
                continue

            # 10th frame: simple sum
            if frame.index == 10:
                frame.score = sum(frame.rolls)
                running_total += frame.score
                continue

            # Strike
            if frame.is_strike():
                idx = roll_pointer[i][0]
                bonus = 0
                if idx + 1 < len(all_rolls):
                    bonus += all_rolls[idx + 1]
                if idx + 2 < len(all_rolls):
                    bonus += all_rolls[idx + 2]
                frame.score = 10 + bonus
                running_total += frame.score
                continue

            # Spare
            if frame.is_spare():
                idx = roll_pointer[i][1]  # second roll index
                bonus = 0
                if idx + 1 < len(all_rolls):
                    bonus += all_rolls[idx + 1]
                frame.score = 10 + bonus
                running_total += frame.score
                continue

            # Open frame
            frame.score = sum(frame.rolls)
            running_total += frame.score

        # Convert to cumulative scores
        cumulative = 0
        for f in self.frames:
            if f.score is not None:
                cumulative += f.score
                f.score = cumulative

    def GetGameState(self):
        """
        Returns a JSON-serializable representation of the game.
        Useful for sending to the frontend.
        """
        return {
            "current_frame": self.CurrentFrame.index,
            "game_over": self.game_over,
            "frames": [
                {
                    "index": f.index,
                    "rolls": f.rolls,
                    "fallen_pins": [list(p) for p in f.fallen_pins],
                    "score": f.score
                }
                for f in self.frames
            ]
        }


# Create a global instance (simple for now)
game_manager = GameManager()
