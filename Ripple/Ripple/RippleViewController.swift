
import Cocoa
import MetalKit

class RippleViewController: NSViewController, MTKViewDelegate {

    var mtkView: MTKView!
    var commandQueue: MTLCommandQueue!
    
    var renderer: Renderer!
    var simulation: Simulation!
    var gridSize = MTLSizeMake(1, 1, 1)
    var waterMesh: MTKMesh!
    var groundTexture: MTLTexture!

    class func buildMesh(device: MTLDevice, gridSize:MTLSize, vertexDescriptor: MTLVertexDescriptor) -> MTKMesh? {
        let metalAllocator = MTKMeshBufferAllocator(device: device)
        
        let mdlMesh = MDLMesh.newPlane(withDimensions: float2(2, 2),
                                       segments: uint2(UInt32(gridSize.width), UInt32(gridSize.height)),
                                       geometryType: .triangles,
                                       allocator: metalAllocator)
        
        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(vertexDescriptor)
        
        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            return nil
        }
        attributes[0].name = MDLVertexAttributePosition
        attributes[1].name = MDLVertexAttributeTextureCoordinate
        
        mdlMesh.vertexDescriptor = mdlVertexDescriptor
        
        return try? MTKMesh(mesh:mdlMesh, device:device)
    }
    
    class func buildVertexDescriptor() -> MTLVertexDescriptor {
        let mtlVertexDescriptor = MTLVertexDescriptor()
        
        mtlVertexDescriptor.attributes[0].format = .float3
        mtlVertexDescriptor.attributes[0].offset = 0
        mtlVertexDescriptor.attributes[0].bufferIndex = 0
        
        mtlVertexDescriptor.attributes[1].format = .float2
        mtlVertexDescriptor.attributes[1].offset = 0
        mtlVertexDescriptor.attributes[1].bufferIndex = 1
        
        mtlVertexDescriptor.layouts[0].stride = 12
        mtlVertexDescriptor.layouts[0].stepRate = 1
        mtlVertexDescriptor.layouts[0].stepFunction = .perVertex
        
        mtlVertexDescriptor.layouts[1].stride = 8
        mtlVertexDescriptor.layouts[1].stepRate = 1
        mtlVertexDescriptor.layouts[1].stepFunction = .perVertex
        
        return mtlVertexDescriptor
    }

    class func loadTexture(device: MTLDevice,
                           textureName: String) throws -> MTLTexture {
        let textureLoader = MTKTextureLoader(device: device)
        
        let textureLoaderOptions = [
            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue)
        ]
        
        return try textureLoader.newTexture(name: textureName,
                                            scaleFactor: 1.0,
                                            bundle: nil,
                                            options: textureLoaderOptions)
        
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let mtkView = self.view as? MTKView else {
            print("View attached to RippleViewController is not an MTKView")
            return
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }
        
        commandQueue = device.makeCommandQueue()!
        
        gridSize = MTLSize(width: 128, height: 128, depth: 1)

        mtkView.device = device
        
        mtkView.depthStencilPixelFormat = .depth32Float_stencil8
        mtkView.colorPixelFormat = .bgra8Unorm_srgb
        mtkView.sampleCount = 1
        mtkView.clearColor = MTLClearColor(red: 0, green: 0.01, blue: 0.04, alpha: 1.0)

        let vertexDescriptor = RippleViewController.buildVertexDescriptor()

        waterMesh = RippleViewController.buildMesh(device: device, gridSize: gridSize, vertexDescriptor: vertexDescriptor)

        do {
            groundTexture = try RippleViewController.loadTexture(device: device, textureName: "cobblestone")
        } catch {
            print("Unable to load texture. Error info: \(error)")
        }

        renderer = Renderer(device: device, vertexDescriptor: vertexDescriptor)

        simulation = Simulation(device: device, gridSize: gridSize)
        
        mtkView.delegate = self
        
        renderer.drawableSize = mtkView.drawableSize
        
        let dropInterval = 0.033
        Timer.scheduledTimer(withTimeInterval: dropInterval, repeats: true) { _ in
            self.addRippleCenter()
        }
    }

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            simulation.writeVertexPositions(to: waterMesh.vertexBuffers.first!.buffer,
                                            gridDimensions: gridSize,
                                            time: now,
                                            commandBuffer: commandBuffer)
            
            if let renderPassDescriptor = view.currentRenderPassDescriptor {
                if let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                    renderer.draw(mesh: waterMesh, texture: groundTexture, commandEncoder: renderEncoder)
                    renderEncoder.endEncoding()
                }
            }
            
            if let drawable = view.currentDrawable {
                commandBuffer.present(drawable)
            }
            
            commandBuffer.commit()
        }
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        renderer.drawableSize = view.drawableSize
    }
    
    func addRippleCenter() {
        let loc = NSMakePoint(CGFloat(drand48() * 2 - 1),
                              CGFloat(drand48() * 2 - 1))
        simulation.addRippleCenter(loc)
    }
    
    override func mouseDown(with event: NSEvent) {
        renderer.wireframe = true
    }
    
    override func mouseUp(with event: NSEvent) {
        renderer.wireframe = false
    }
}
