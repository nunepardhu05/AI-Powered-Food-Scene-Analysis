from typing import Literal
from pydantic import BaseModel, Field, model_validator

class Target(BaseModel):
    minimum: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    maximum: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    unit: str
    source: str = 'clinician-entered'

    @model_validator(mode='after')
    def ordered(self):
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError('Target minimum must not exceed maximum')
        return self

class Profile(BaseModel):
    id: str = Field(default='local', pattern=r'^[A-Za-z0-9_-]{1,64}$')
    name: str = Field(default='', max_length=100)
    age: int = Field(default=30, ge=0, le=120)
    sex: Literal['female', 'male', 'other', 'unspecified'] = 'unspecified'
    height_cm: float = Field(default=170, ge=30, le=250, allow_inf_nan=False)
    weight_kg: float = Field(default=70, ge=1, le=500, allow_inf_nan=False)
    waist_cm: float | None = Field(default=None, gt=0, le=300, allow_inf_nan=False)
    activity: Literal['sedentary', 'light', 'moderate', 'active'] = 'light'
    goal: Literal['maintain', 'lose', 'gain'] = 'maintain'
    region: str = 'Indian'
    diet: Literal['omnivore', 'vegetarian', 'vegan'] = 'vegetarian'
    restrictions: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    clinical: dict[str, float] = Field(default_factory=dict)
    medications: list[str] = Field(default_factory=list)
    targets: dict[str, Target] = Field(default_factory=dict)
    meal_targets: dict[str, Target] = Field(default_factory=dict)
    pregnancy_trimester: int | None = Field(default=None, ge=1, le=3)

class FoodEntry(BaseModel):
    food_id: str
    grams: float = Field(gt=0, le=5000, allow_inf_nan=False)
    uncertainty: float = Field(default=0.1, ge=0, le=1, allow_inf_nan=False)
    basis: Literal['measured', 'user-estimate', 'model'] = 'user-estimate'

class MealRequest(BaseModel):
    profile_id: str = 'local'
    items: list[FoodEntry] = Field(min_length=1, max_length=50)
    slot: Literal['breakfast','mid-morning snack','lunch','evening snack','dinner'] = 'lunch'
    percentages: list[float] = Field(default_factory=lambda:[100.0], min_length=1, max_length=100)

    @model_validator(mode='after')
    def portions(self):
        import math
        if any(not math.isfinite(x) or x <= 0 for x in self.percentages) or abs(sum(self.percentages)-100)>0.001:
            raise ValueError('Positive serving percentages must sum to 100')
        return self
