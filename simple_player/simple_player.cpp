#include <stdio.h>
#include <memory>
#include <vector>
#include <queue>
#include <chrono>

//OpenVINO including
#include <inference_engine.hpp>
#if defined(_WIN32) || defined(_WIN64) 
#include <gpu/gpu_context_api_dx.hpp>
#else //LINUX
#include <gpu/gpu_context_api_va.hpp>
#endif
#include <cldnn/cldnn_config.hpp>
#include "common.hpp"

//Media SDK including
#include "sample_utils.h"
#include "sample_defs.h"
#include "decode_render.h"
#include "general_allocator.h"
#include "opencl_filter.h"
#if defined(_WIN32) || defined(_WIN64) 
#include "d3d_device.h"
#include "d3d_allocator.h"
#include "d3d11_device.h"
#include "d3d11_allocator.h"
#include "opencl_filter_dx11.h"
#else //LINUX
#include "vaapi_allocator.h"
#include "vaapi_device.h"
#include "vaapi_utils.h"
#include "opencl_filter_va.h"
#endif

//
using namespace std::chrono;
using namespace InferenceEngine;

#define OW 1920
#define OH 1088
#define IW 960
#define IH 544


class simple_player {
public:
    simple_player();
    virtual ~simple_player() {};
    mfxStatus InitDecoderRender(CSmplBitstreamReader* pFileReader);
    mfxStatus CreateRenderingWindow();

    int InitOpenVINO(char* saved_model);
    //int LoadModel();
    int Decoding();
    void Close();

private:
    //Media SDK
    MFXVideoSession     m_mfxSession;
    GeneralAllocator*   m_pGeneralAllocator;
#if defined(_WIN32) || defined(_WIN64) 
    CDecodeD3DRender*   m_pD3dRender;
#endif
    CHWDevice*          m_hwdev;
    MFXVideoDECODE*     m_pmfxDEC;

    CSmplBitstreamReader* m_pFileReader;

    mfxBitstreamWrapper  m_mfxBS; // contains encoded data
    OpenCLFilter* m_pOpenCLFilter;
    //Allocate and free surface
    mfxFrameAllocResponse m_mfxRenderResponse;
    mfxFrameAllocResponse m_mfxDecResponse;
    std::vector<mfxFrameSurface1> m_decoderSurf;
    std::vector<mfxFrameSurface1> m_renderSurf;
    

    //OpenVINO
    //InferenceEngine
    Core *m_povCore;
    InferRequest m_ovInfReq;

    std::string m_ovInName;
    std::string m_ovOutName;
#if defined(_WIN32) || defined(_WIN64) 
    gpu::D3DContext::Ptr m_povGPUCtx;
#else
    gpu::VAContext::Ptr  m_povGPUCtx;
#endif
    Blob::Ptr m_povOutBlob;
    cl::Buffer m_ovOutBuffer;
};

simple_player::simple_player() {

}

mfxStatus simple_player::InitDecoderRender(CSmplBitstreamReader* pFileReader) {

    mfxStatus sts = MFX_ERR_NONE;
    mfxHandleType hdl_t;
    mfxHDL hdl = NULL;
    
    //
    //Initialize Media SDK session
    //
    mfxInitParamlWrap initPar;
    initPar.Version.Major = 1;
    initPar.Version.Minor = 0;
    initPar.GPUCopy = 1; //True
#if defined(_WIN32) || defined(_WIN64) 
    initPar.Implementation = MFX_IMPL_HARDWARE | MFX_IMPL_VIA_D3D11;
    hdl_t = MFX_HANDLE_D3D11_DEVICE;
#else
    initPar.Implementation = MFX_IMPL_HARDWARE | MFX_IMPL_VIA_VAAPI;
    hdl_t = MFX_HANDLE_VA_DISPLAY;
#endif

    sts = m_mfxSession.InitEx(initPar);
    MSDK_CHECK_STATUS(sts, "m_mfxSession.QueryVersion failed");

    mfxVersion version;
    sts = m_mfxSession.QueryVersion(&version); // get real API version of the loaded library
    MSDK_CHECK_STATUS(sts, "m_mfxSession.QueryVersion failed");
    std::cout << "Media SDK version: " << version.Major << "." << version.Minor << std::endl;
    
    //
    //Initialize hardware device
    //
    
#if defined(_WIN32) || defined(_WIN64) 
    m_hwdev = new CD3D11Device();    
    sts = m_hwdev->Init(NULL, 1, MSDKAdapter::GetNumber(m_mfxSession));
    MSDK_CHECK_STATUS(sts, "hwdev->Init failed");
#else
    int type = MFX_LIBVA_X11;
    std::string device("");
    mfxI32 monitorType = 0;
    m_hwdev = CreateVAAPIDevice(device, type);
    //sts = m_hwdev->Init(&monitorType, 1, MSDKAdapter::GetNumber(m_mfxSession));
    //no window debug
    sts = m_hwdev->Init(&monitorType, 0, MSDKAdapter::GetNumber(m_mfxSession));
    MSDK_CHECK_STATUS(sts, "hwdev->Init failed");
#endif
    
    //
    //Initialize render
    //
#if defined(_WIN32) || defined(_WIN64) 
    m_pD3dRender = new CDecodeD3DRender();
    m_pD3dRender->SetHWDevice(m_hwdev);
#endif

    sts = m_hwdev->GetHandle(hdl_t, &hdl);
    MSDK_CHECK_STATUS(sts, "hwdev->GetHandle failed");

    sts = m_mfxSession.SetHandle(hdl_t, hdl);
    MSDK_CHECK_STATUS(sts, "m_mfxSession.SetHandle failed");

    //Create Video Surface Allocator
    m_pGeneralAllocator = new GeneralAllocator();
    sts = m_mfxSession.SetFrameAllocator(m_pGeneralAllocator);
    MSDK_CHECK_STATUS(sts, "m_mfxSession.SetFrameAllocator failed");

#if defined(_WIN32) || defined(_WIN64) 
    mfxAllocatorParams* pAllocParam(new D3D11AllocatorParams());
    D3D11AllocatorParams* pD3D11Params = dynamic_cast<D3D11AllocatorParams*>(pAllocParam);
    pD3D11Params->pDevice = reinterpret_cast<ID3D11Device*>(hdl); 
    sts = m_pGeneralAllocator->Init(pD3D11Params);
    MSDK_CHECK_STATUS(sts, "m_pGeneralAllocator->Init failed");
#else
    VADisplay va_dpy = NULL;
    sts = m_hwdev->GetHandle(MFX_HANDLE_VA_DISPLAY, (mfxHDL*)&va_dpy);
    MSDK_CHECK_STATUS(sts, "m_hwdev->GetHandle failed");
    vaapiAllocatorParams* p_vaapiAllocParams = new vaapiAllocatorParams;
    MSDK_CHECK_POINTER(p_vaapiAllocParams, MFX_ERR_MEMORY_ALLOC);

    p_vaapiAllocParams->m_dpy = va_dpy;
    //MODE_RENDERING 
    p_vaapiAllocParams->m_export_mode = vaapiAllocatorParams::PRIME;
    int m_export_mode = p_vaapiAllocParams->m_export_mode;
    std::cout << "m_export_mode = " << m_export_mode << std::endl;
    // initialize memory allocator
    sts = m_pGeneralAllocator->Init(p_vaapiAllocParams);
    MSDK_CHECK_STATUS(sts, "m_pGeneralAllocator->Init failed");
#endif

    //
    //Create decoder and parse codec info.
    //
    MfxVideoParamsWrapper   mfxDecVideoParams;
    m_pmfxDEC = new MFXVideoDECODE(m_mfxSession);
    MSDK_CHECK_POINTER(m_pmfxDEC, MFX_ERR_MEMORY_ALLOC);

    memset(&mfxDecVideoParams, 0, sizeof(mfxDecVideoParams));
    mfxDecVideoParams.mfx.CodecId = MFX_CODEC_AVC;
    mfxDecVideoParams.IOPattern = MFX_IOPATTERN_OUT_VIDEO_MEMORY;

    //Read bit stream and parse codec info
    m_pFileReader = pFileReader;
    m_mfxBS.Extend(4 * 1024 * 1024);
    sts = pFileReader->ReadNextFrame(&m_mfxBS);
    MSDK_CHECK_STATUS(sts, "pFileReader->ReadNextFrame failed");

    //Parse codec info
    m_pmfxDEC->DecodeHeader(&m_mfxBS, &mfxDecVideoParams);
    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

    mfxFrameAllocRequest DecRequest;
    memset(&DecRequest, 0, sizeof(DecRequest));
    sts = m_pmfxDEC->QueryIOSurf(&mfxDecVideoParams, &DecRequest);
    MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
   
    //set up mfxFrameAllocRequest for rendering buffer
    mfxFrameAllocRequest RenderRequest;
    memset(&RenderRequest, 0, sizeof(mfxFrameAllocRequest));
    //RenderRequest.Type = 0x82a;//0x0822;//MFX_MEMTYPE_FROM_VPPOUT MFX_MEMTYPE_EXPORT_FRAME MFX_MEMTYPE_EXTERNAL_FRAME
    RenderRequest.Type = MFX_MEMTYPE_FROM_VPPOUT | MFX_MEMTYPE_DXVA2_PROCESSOR_TARGET |
        MFX_MEMTYPE_EXPORT_FRAME | MFX_MEMTYPE_EXTERNAL_FRAME;
    RenderRequest.NumFrameMin = 2;
    RenderRequest.NumFrameSuggested = 2;
    RenderRequest.Info.FourCC = MFX_FOURCC_RGB4;
    RenderRequest.Info.CropW = OW;
    RenderRequest.Info.CropH = OH;
    RenderRequest.Info.Width = OW;
    RenderRequest.Info.Height = OH; 
    
    // Determine the required number of surfaces for decoder output (VPP input) and for VPP output
    mfxU16 nSurfNumDecVPP = DecRequest.NumFrameSuggested;
    mfxU16 nSurfNumPluginOut = 2;

    //
    // Allocate surfaces for decoder and render
    //
    //mfxFrameAllocResponse mfxDecResponse;
    memset(&m_mfxDecResponse, 0, sizeof(mfxFrameAllocResponse));
    sts = m_pGeneralAllocator->Alloc(m_pGeneralAllocator->pthis, &DecRequest, &m_mfxDecResponse);
    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
    m_decoderSurf.resize(nSurfNumDecVPP);
    for (int i = 0; i < nSurfNumDecVPP; i++) {
        memset(&m_decoderSurf[i], 0, sizeof(mfxFrameSurface1));
        //pmfxSurfaces[i].Info = mfxDecVideoParams.mfx.FrameInfo;
        MSDK_MEMCPY_VAR(m_decoderSurf[i].Info, &(DecRequest.Info), sizeof(mfxFrameInfo));
        m_decoderSurf[i].Data.MemId = m_mfxDecResponse.mids[i];      // MID (memory id) represents one video NV12 surface
    }

    //mfxFrameAllocResponse mfxRenderResponse;
    memset(&m_mfxRenderResponse, 0, sizeof(mfxFrameAllocResponse));
    sts = m_pGeneralAllocator->Alloc(m_pGeneralAllocator->pthis, &RenderRequest, &m_mfxRenderResponse);
    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
    m_renderSurf.resize(nSurfNumPluginOut);
    for (int i = 0; i < nSurfNumPluginOut; i++) {
        memset(&m_renderSurf[i], 0, sizeof(mfxFrameSurface1));
        MSDK_MEMCPY_VAR(m_renderSurf[i].Info, &(RenderRequest.Info), sizeof(mfxFrameInfo));
        m_renderSurf[i].Data.MemId = m_mfxRenderResponse.mids[i];    // MID (memory id) represent one D3D NV12 surface
    }

    //
    //Create render target window
    //
    sts = CreateRenderingWindow();
    MSDK_CHECK_STATUS(sts, "CreateRenderingWindow failed");

    //
    // Initialize decoder
    //
    sts = m_pmfxDEC->Init(&mfxDecVideoParams);
    MSDK_IGNORE_MFX_STS(sts, MFX_WRN_PARTIAL_ACCELERATION);
    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

    //
    //Initialize an OpenCL filter for post process before render output
    //
    cl_int error = CL_SUCCESS;
#if defined(_WIN32) || defined(_WIN64) 
    m_pOpenCLFilter = new OpenCLFilterDX11();
#else
    m_pOpenCLFilter = new OpenCLFilterVA();
#endif
    error = m_pOpenCLFilter->AddKernel(readFile("convert.cl").c_str(), "StoreImage");
    if (error) return MFX_ERR_DEVICE_FAILED;

    error = m_pOpenCLFilter->OCLInit(hdl);//no use input parameter

    error = m_pOpenCLFilter->SelectKernel(0);
    if (error) return MFX_ERR_DEVICE_FAILED;

    return sts;
}

int simple_player::InitOpenVINO(char* saved_model) {

    mfxStatus sts = MFX_ERR_NONE;
    const char* device = "GPU";

    //
    // --------------------------- 1. Load inference engine -------------------------------------
    //
    m_povCore = new InferenceEngine::Core();
    std::cout << m_povCore->GetVersions(device) << std::endl;

    //
    // 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) forma
    //
    CNNNetwork network = m_povCore->ReadNetwork(saved_model);
    if (network.getOutputsInfo().size() != 1) throw std::logic_error("Sample supports topologies with 1 output only");
    if (network.getInputsInfo().size() != 1) throw std::logic_error("Sample supports topologies with 1 input only");
   
    //
    // --------------------------- 3. Configure input & output ---------------------------------------------
    //
    // --------------------------- Prepare input blobs -----------------------------------------------------
     /** Taking information about all topology inputs **/
    InputsDataMap inputInfo(network.getInputsInfo());
    auto inputInfoItem = *inputInfo.begin();
    //inputInfoItem.second->setPrecision(Precision::FP32);

    inputInfoItem.second->setLayout(InferenceEngine::Layout::NCHW);
    inputInfoItem.second->setPrecision(Precision::U8);
    inputInfoItem.second->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::NV12);

    m_ovInName = inputInfoItem.second->name();
    //int n_ch = inputInfoItem.second->getPreProcess().getNumberOfChannels();
    //std::cout << "getNumberOfChannels: " << n_ch << std::endl;

    OutputsDataMap outputInfo(network.getOutputsInfo());
    // BlobMap outputBlobs;
    //std::string firstOutputName;
    auto outputInfoItem = *outputInfo.begin();
    for (auto& item : outputInfo) {
        if (m_ovOutName.empty()) {
            m_ovOutName = item.first;
        }
        DataPtr outputData = item.second;
        if (!outputData) {
            throw std::logic_error("output data pointer is not valid");
        }

        item.second->setPrecision(Precision::FP32);
    }

#if defined(_WIN32) || defined(_WIN64) 
    CComPtr<ID3D11Device>   pD3D11Device;
    mfxHDL displayHandle = { 0 };
    sts = m_mfxSession.GetHandle(MFX_HANDLE_D3D11_DEVICE, &displayHandle);
    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);
    pD3D11Device = (ID3D11Device*)displayHandle;
    m_povGPUCtx = gpu::make_shared_context(*m_povCore, device, pD3D11Device);
    std::cout << "make_shared_context " << m_povGPUCtx << std::endl;
#else
    VADisplay va_dpy = NULL;
    sts = m_hwdev->GetHandle(MFX_HANDLE_VA_DISPLAY, (mfxHDL*)&va_dpy);
    printf("disp = 0x %x\n", va_dpy);

    // create the shared context object
    m_povGPUCtx = gpu::make_shared_context(*m_povCore, device, va_dpy);
    printf("shared_gpu_context = 0x%x\n", m_povGPUCtx);
#endif

    //
    // --------------------------- 4. Loading model to the device ------------------------------------------
    //
    std::cout << "Load network " << saved_model << std::endl;
    // compile network within a shared context
    ExecutableNetwork executable_network = m_povCore->LoadNetwork(network, m_povGPUCtx,
        { { CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS,PluginConfigParams::YES } });
    

    //
    // --------------------------- 5. Create infer request -------------------------------------------------
    //
    std::cout << "Create infer request" << std::endl;
    //InferRequest infer_request = executable_network.CreateInferRequest();
    m_ovInfReq = executable_network.CreateInferRequest();
    // -----------------------------------------------------------------------------------------------------
    for (int i = 0; i < 4; i++) std::cout << "DIM " << inputInfoItem.second->getTensorDesc().getDims()[i] << std::endl;
    std::cout << inputInfoItem.second->name() << std::endl;

    //
    //----------------Prepare share buffer with OpenCL----------------------
    //
    std::cout << "Share with OpenCL" << std::endl;
    auto cldnn_context = executable_network.GetContext();
    cl_context ctx = std::dynamic_pointer_cast<gpu::ClContext>(cldnn_context)->get();
    cl::Context _context(ctx, true);
    cl::Device _device = cl::Device(_context.getInfo<CL_CONTEXT_DEVICES>()[0].get(), true);
    cl::CommandQueue _queue;
    cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    _queue = cl::CommandQueue(_context, _device, props);

    //inference output buffer, scale x2
    m_ovOutBuffer = cl::Buffer(_context, CL_MEM_READ_WRITE, OW * OH * 3 * 4);
    // wrap the buffer into RemoteBlob
    m_povOutBlob = gpu::make_shared_blob(outputInfoItem.second->getTensorDesc(),
        cldnn_context, m_ovOutBuffer);

    return 0;
}

int GetFreeSurfaceIndex(const std::vector<mfxFrameSurface1>& pSurfacesPool)
{
    auto it = std::find_if(pSurfacesPool.begin(), pSurfacesPool.end(), [](const mfxFrameSurface1& surface) {
        return 0 == surface.Data.Locked;
        });

    if (it == pSurfacesPool.end())
        return MFX_ERR_NOT_FOUND;
    else return it - pSurfacesPool.begin();
}

mfxStatus simple_player::CreateRenderingWindow()
{
    mfxStatus sts = MFX_ERR_NONE;
    bool bVppIsUsed = true;    

#if D3D_SURFACES_SUPPORT
    sWindowParams windowParams;

    windowParams.lpWindowName = MSDK_STRING("sample_decode");
    windowParams.nx = 0;
    windowParams.ny = 0;
    if (bVppIsUsed)
    {
        windowParams.nWidth = OW / 2;
        windowParams.nHeight = OH / 2;
    }
    else
    {
        windowParams.nWidth = OW;
        windowParams.nHeight = OH;
    }

    windowParams.ncell = 0;
    windowParams.nAdapter = 0;

    windowParams.lpClassName = MSDK_STRING("Render Window Class");
    windowParams.dwStyle = WS_OVERLAPPEDWINDOW;
    windowParams.hWndParent = NULL;
    windowParams.hMenu = NULL;
    windowParams.hInstance = GetModuleHandle(NULL);
    windowParams.lpParam = NULL;
    windowParams.bFullScreen = FALSE;

    sts = m_pD3dRender->Init(windowParams);
    MSDK_CHECK_STATUS(sts, "m_d3dRender.Init failed");

#endif
    return sts;
}


int simple_player::Decoding() {

    mfxStatus sts = MFX_ERR_NONE;
    mfxSyncPoint syncpD = NULL;
    mfxSyncPoint syncpV = NULL;
    mfxSyncPoint syncpP = NULL; //for plugin
    mfxFrameSurface1* pmfxOutSurface = NULL;//decode out, VPP in

    int nIndex = 0;
    int nIndex2 = 0;
    int nIndex3 = 0;
    mfxU32 nFrame = 0;

    mfxBitstream* pBitstream = &m_mfxBS;

    //
    // Stage 1: Main decoding loop
    //
    while (MFX_ERR_NONE <= sts || MFX_ERR_MORE_DATA == sts || MFX_ERR_MORE_SURFACE == sts) {
        if (MFX_WRN_DEVICE_BUSY == sts)
            MSDK_SLEEP(1);  // Wait if device is busy, then repeat the same call to DecodeFrameAsync

        if (MFX_ERR_MORE_DATA == sts) {
            sts = m_pFileReader->ReadNextFrame(pBitstream);      // Read more data into input bit stream
            MSDK_BREAK_ON_ERROR(sts);
        }

        if (MFX_ERR_MORE_SURFACE == sts || MFX_ERR_NONE == sts) {
            nIndex = GetFreeSurfaceIndex(m_decoderSurf);     // Find free frame surface
            MSDK_CHECK_ERROR(MFX_ERR_NOT_FOUND, nIndex, MFX_ERR_MEMORY_ALLOC);
        }
        // Decode a frame asychronously (returns immediately)
        sts = m_pmfxDEC->DecodeFrameAsync(&m_mfxBS, &m_decoderSurf[nIndex], &pmfxOutSurface, &syncpD);

        // Ignore warnings if output is available,
        // if no output and no action required just repeat the DecodeFrameAsync call
        if (MFX_ERR_NONE < sts && syncpD)
            sts = MFX_ERR_NONE;

        if (MFX_ERR_NONE == sts)
            sts = m_mfxSession.SyncOperation(syncpD, 60000);     // Synchronize. Wait until decoded frame is ready

        if (MFX_ERR_NONE == sts) {
            
            ++nFrame;

            //---------------------------------
#if defined(_WIN32) || defined(_WIN64) 
            mfxHDLPair mid_pair = { 0 };

            sts = m_pGeneralAllocator->GetHDL(m_pGeneralAllocator->pthis,
                pmfxOutSurface->Data.MemId, (mfxHDL*)&mid_pair);
            MSDK_CHECK_STATUS(sts, "pAlloc->GetHDL failed");

            ID3D11Texture2D* surface = (ID3D11Texture2D*)mid_pair.first;
            auto shared_blob = gpu::make_shared_blob_nv12(IH, IW, m_povGPUCtx, surface);
#else
            VASurfaceID* surface = NULL;
            printf("Get surface id from %x\n", pmfxOutSurface->Data.MemId);
            sts = m_pGeneralAllocator->GetHDL(m_pGeneralAllocator->pthis, pmfxOutSurface->Data.MemId,
                reinterpret_cast<mfxHDL*>(&surface));
            MSDK_CHECK_STATUS(sts, "pAlloc->GetFrameHDL failed");
            printf("Get a surface:%x,  %x\n", pmfxOutSurface->Data.MemId, *surface);
            auto shared_blob = gpu::make_shared_blob_nv12(IH, IW, m_povGPUCtx, *surface);
#endif            
            //make output blob

            m_ovInfReq.SetBlob(m_ovInName, shared_blob);
            //set output blob
            m_ovInfReq.SetBlob(m_ovOutName, m_povOutBlob);

            // --------------------------- 7. Do inference ---------------------------------------------------------
            std::cout << "Start inference" << std::endl;
            high_resolution_clock::time_point start = high_resolution_clock::now();

            m_ovInfReq.Infer();

            high_resolution_clock::time_point now = high_resolution_clock::now();
            duration<double, std::milli> time_span = now - start;
            std::cout << "Infer took " << time_span.count() << " milliseconds.";
            std::cout << std::endl;
            // --------------------------- 8. Process output -------------------------------------------------------
            //check inference output
#if 0
            int numElements = OW * OH * 3;
            std::vector<float> output(numElements, 0);
            cl::copy(m_ovOutBuffer, begin(output), end(output));
#endif
            cl_mem infer_out = m_ovOutBuffer.get();


            int dst_w = 1920;
            int dst_h = 1088;
            //
            m_pOpenCLFilter->SetAllocator(m_pGeneralAllocator);
            nIndex3 = GetFreeSurfaceIndex(m_renderSurf);//todo: lock?
            cl_int error = m_pOpenCLFilter->ProcessSurface(dst_w, dst_h,
                infer_out, m_renderSurf[nIndex3].Data.MemId);

            

            //Aslan:window message
#if defined(_WIN32) || defined(_WIN64) 
            m_pD3dRender->UpdateTitle(30);
            int res = m_pD3dRender->RenderFrame(&m_renderSurf[nIndex3], m_pGeneralAllocator);
#else
            m_hwdev->UpdateTitle(30);
            sts = m_hwdev->RenderFrame(&m_renderSurf[nIndex3], m_pGeneralAllocator);
#endif
            MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

        }
    }

    // MFX_ERR_MORE_DATA means that file has ended, need to go to buffering loop, exit in case of other errors
    MSDK_IGNORE_MFX_STS(sts, MFX_ERR_MORE_DATA);
    MSDK_CHECK_RESULT(sts, MFX_ERR_NONE, sts);

    //
    // Ignore following stage to clean residual fram in pipeline
    // Stage 2: Retrieve the buffered decoded frames
    // Stage 3: Retrieve the buffered VPP frames
    //

    // ===================================================================
    // Clean up resources
    //  - It is recommended to close Media SDK components first, before releasing allocated surfaces, since
    //    some surfaces may still be locked by internal Media SDK resources.

    m_pmfxDEC->Close();

}

void simple_player::Close() {

    if (m_pmfxDEC) {
        m_pmfxDEC->Close();
        delete m_pmfxDEC;
    }

}

#if defined(_WIN32) || defined(_WIN64) 
int _tmain(int argc, TCHAR* argv[])
#else
int main(int argc, char* argv[])
#endif
{
    mfxStatus sts = MFX_ERR_NONE;
    bool bEnableOutput = false; // if true, removes all YUV file writing

#if defined(_WIN32) || defined(_WIN64) 
    #define BUFFER_SIZE 100
    char input_model[BUFFER_SIZE];
    char output_file[BUFFER_SIZE];
    TCHAR* input_file = argv[1];
    
    size_t    n = 0;
    wcstombs_s(&n, input_model, BUFFER_SIZE, argv[3], BUFFER_SIZE);
    wcstombs_s(&n, output_file, BUFFER_SIZE, argv[2], BUFFER_SIZE);
#else
    char *input_model = "/home/aslan/workspace/openvino_model/rrdn_544x960_rgb_255/saved_model.xml";
    char *output_file = "sr_out.yuv";
    char *input_file ="/home/aslan/workspace/csgo_960x540_30fps_4M.h264";
#endif

    //Initialize source file reader
    CSmplBitstreamReader* pFileReader = new CH264FrameReader();
    sts = pFileReader->Init(input_file);
    MSDK_CHECK_STATUS(sts, "m_FileReader->Init failed");

    simple_player sr_player;
    sr_player.InitDecoderRender(pFileReader);
    
    sr_player.InitOpenVINO(input_model);

    sr_player.Decoding();

    return 0;
}