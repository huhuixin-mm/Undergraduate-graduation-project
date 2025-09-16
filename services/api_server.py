"""
FastAPI server for Meditron3-Qwen2.5 model
Provides REST API endpoints for model inference
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import json
from pathlib import Path
import sys
import uvicorn
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings, get_project_root
from services.model_handler import get_model


# Pydantic models for request/response
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    max_new_tokens: int = Field(512, ge=1, le=2048, description="Maximum new tokens to generate")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(50, ge=1, le=200, description="Top-k sampling")
    do_sample: bool = Field(True, description="Whether to use sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    stream: bool = Field(False, description="Whether to stream the response")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    max_new_tokens: int = Field(512, ge=1, le=2048, description="Maximum new tokens to generate")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling")
    stream: bool = Field(False, description="Whether to stream the response")


class GenerationResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    prompt: str = Field(..., description="Original prompt")
    generation_time: float = Field(..., description="Generation time in seconds")


class ModelInfoResponse(BaseModel):
    status: str = Field(..., description="Model status")
    model_path: Optional[str] = Field(None, description="Model path")
    device: Optional[str] = Field(None, description="Device used")
    vocab_size: Optional[int] = Field(None, description="Vocabulary size")
    gpu_name: Optional[str] = Field(None, description="GPU name if available")


# Initialize FastAPI app
app = FastAPI(
    title="Meditron3-Qwen2.5 API",
    description="REST API for Meditron3-Qwen2.5 medical language model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_loaded = False


def setup_logging():
    """Setup logging configuration"""
    log_dir = get_project_root() / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "api.log",
        level=settings.logging.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB"
    )


async def load_model_async():
    """Load model asynchronously"""
    global model, model_loaded
    
    logger.info("🚀 Starting model loading...")
    
    try:
        model = get_model()
        success = await asyncio.get_event_loop().run_in_executor(
            None, model.load_model
        )
        
        if success:
            model_loaded = True
            logger.info("✅ Model loaded successfully")
        else:
            logger.error("❌ Failed to load model")
            raise RuntimeError("Model loading failed")
            
    except Exception as e:
        logger.error(f"❌ Model loading error: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    setup_logging()
    logger.info("🌟 Starting Meditron3-Qwen2.5 API server")
    
    try:
        await load_model_async()
        logger.info("🎉 Server startup completed")
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global model, model_loaded
    
    logger.info("🔄 Shutting down server...")
    
    if model and model_loaded:
        model.unload_model()
        model_loaded = False
        logger.info("🗑️ Model unloaded")
    
    logger.info("👋 Server shutdown completed")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Meditron3-Qwen2.5 API Server",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_loaded
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "timestamp": str(asyncio.get_event_loop().time())
    }


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        info = model.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"❌ Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text from prompt"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if request.stream:
        # For streaming, use the streaming endpoint
        raise HTTPException(
            status_code=400, 
            detail="Use /generate/stream endpoint for streaming responses"
        )
    
    try:
        import time
        start_time = time.time()
        
        # Generate response
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            model.generate_response,
            request.prompt,
            request.max_new_tokens,
            request.temperature,
            request.top_p,
            request.top_k,
            request.do_sample,
            request.repetition_penalty,
            False  # stream=False
        )
        
        generation_time = time.time() - start_time
        
        logger.info(f"💬 Generated response in {generation_time:.2f}s")
        
        return GenerationResponse(
            response=response,
            prompt=request.prompt,
            generation_time=generation_time
        )
        
    except Exception as e:
        logger.error(f"❌ Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/stream")
async def generate_text_stream(request: GenerationRequest):
    """Generate text with streaming response"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    async def stream_response():
        try:
            # Run streaming generation in executor
            generator = await asyncio.get_event_loop().run_in_executor(
                None,
                model.generate_response,
                request.prompt,
                request.max_new_tokens,
                request.temperature,
                request.top_p,
                request.top_k,
                request.do_sample,
                request.repetition_penalty,
                True  # stream=True
            )
            
            for token in generator:
                # Format as Server-Sent Events
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"
            
            # Send end signal
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"❌ Streaming error: {str(e)}")
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/chat")
async def chat_completion(request: ChatRequest):
    """Chat completion endpoint"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Convert messages to the format expected by model
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        # Generate response
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            model.chat,
            messages,
            # Pass additional kwargs
            {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": request.stream
            }
        )
        
        generation_time = time.time() - start_time
        
        logger.info(f"💬 Chat response generated in {generation_time:.2f}s")
        
        return {
            "response": response,
            "generation_time": generation_time
        }
        
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/reload")
async def reload_model():
    """Reload the model (admin endpoint)"""
    global model_loaded
    
    try:
        logger.info("🔄 Reloading model...")
        
        if model and model_loaded:
            model.unload_model()
            model_loaded = False
        
        await load_model_async()
        
        return {"status": "success", "message": "Model reloaded successfully"}
        
    except Exception as e:
        logger.error(f"❌ Model reload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main function to run the server"""
    logger.info("🚀 Starting Meditron3-Qwen2.5 API server")
    
    uvicorn.run(
        "services.api_server:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        reload=settings.api.reload,
        log_level=settings.logging.log_level.lower()
    )


if __name__ == "__main__":
    main()