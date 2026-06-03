"""Data transfer objects and schemas for the Shipping Email Classification API.

This module defines the request and response schemas, as well as the
structured data extraction schemas for each email category (tonnage,
voyage charter cargo, and time charter cargo).
"""

from __future__ import annotations

from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field


class ParseEmailRequest(BaseModel):
    """Schema for incoming email parsing requests."""
    
    email_body: str = Field(
        ..., 
        description="The raw multi-line email text to classify and parse."
    )


class TonnageRecord(BaseModel):
    """Structured record representing a vessel availability (tonnage) entry."""
    
    vessel_name: str = Field(
        default="", 
        description="The name of the vessel (e.g., MV SHENG AN HAI)."
    )
    account_name: str = Field(
        default="", 
        description="The name of the company or broker offering the vessel."
    )
    open_port: str = Field(
        default="", 
        description="The port where the vessel becomes available."
    )
    open_date: str = Field(
        default="", 
        description="The date when the vessel becomes available (normalized format)."
    )
    vessel_type: str = Field(
        default="", 
        description="The structural type or class of the vessel (e.g., SUPRAMAX)."
    )
    vessel_size: str = Field(
        default="", 
        description="The deadweight tonnage (DWT) size of the vessel as a numeric string."
    )


class CargoVCRecord(BaseModel):
    """Structured record representing a voyage charter (VC) cargo requirement."""
    
    account_name: str = Field(
        default="", 
        description="The name of the company or charterer offering the cargo."
    )
    cargo_name: str = Field(
        default="", 
        description="The name of the commodity/cargo (e.g., COAL, HRC)."
    )
    loading_port: str = Field(
        default="", 
        description="The port of loading (POL)."
    )
    discharge_port: str = Field(
        default="", 
        description="The port of discharge (POD)."
    )
    laycan: str = Field(
        default="", 
        description="The layday/canceling window (normalized format)."
    )
    cargo_type: str = Field(
        default="", 
        description="The category of the cargo (e.g., BULK, TANKER)."
    )


class CargoTCRecord(BaseModel):
    """Structured record representing a time charter (TC) requirement."""
    
    account_name: str = Field(
        default="", 
        description="The name of the company or charterer offering the cargo."
    )
    cargo_name: str = Field(
        default="", 
        description="The name of the commodity/cargo (e.g., GRAIN, STEELS)."
    )
    delivery_port: str = Field(
        default="", 
        description="The port/area where the vessel is delivered to the charterer."
    )
    redelivery_port: str = Field(
        default="", 
        description="The port/area where the vessel is redelivered to the owner."
    )
    duration: str = Field(
        default="", 
        description="The duration or period of the time charter (e.g., ABT 30 DAYS)."
    )
    laycan: str = Field(
        default="", 
        description="The layday/canceling window (normalized format)."
    )
    cargo_type: str = Field(
        default="", 
        description="The category of the cargo (e.g., BULK, GENERAL)."
    )


class ResponseMetadata(BaseModel):
    """Execution metadata associated with a parse request."""
    
    records_found: int = Field(
        default=0, 
        description="The total number of structured records extracted from the email."
    )
    processing_time_ms: float = Field(
        default=0.0, 
        description="The server-side processing duration in milliseconds."
    )


class ParseEmailResponse(BaseModel):
    """Unified response schema for email parsing requests."""
    
    success: bool = Field(
        default=True, 
        description="Indicates whether the parsing request was successful."
    )
    category: str = Field(
        default="", 
        description="The classified email category: 'tonnage', 'cargo_vc', or 'cargo_tc'."
    )
    confidence: float = Field(
        default=0.0, 
        description="The classification confidence score in the range [0.0, 1.0]."
    )
    records: List[Union[TonnageRecord, CargoVCRecord, CargoTCRecord, Dict[str, Any]]] = Field(
        default_factory=list,
        description="The list of structured records extracted from the email text."
    )
    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata,
        description="The performance and execution metadata."
    )


class HealthResponse(BaseModel):
    """Health check status response schema."""
    
    status: str = Field(
        default="healthy", 
        description="The health status of the application."
    )
