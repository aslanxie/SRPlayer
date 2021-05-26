constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

//a sample code without performance consideration
//load R, G, B, A interleaved image to R, G and B seperate plane
__kernel void LoadImage(read_only image2d_t imageA, global float* output)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int w = get_global_size(0);
	const int h = get_global_size(1);

	float4 A = read_imagef(imageA, sampler, (int2)(x, y));
	//uint4 B = read_imageui(imageB, sampler, (int2)(x, y));
	output[y * w + x] = A.x;
	output[w * h + y * w + x] = A.y;
	output[2 * w * h + y * w + x] = A.z;
}

//output buffer
__kernel void StoreImage(global float* output, write_only image2d_t imageB)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int w = get_global_size(0);
	const int h = get_global_size(1);
	//if(x > 1910 && y > 1080) printf("w = %d, h = %d\n",x, y);
	//float4 A = read_imagef(imageA, sampler, (int2)(x, y));
	//uint4 B = read_imageui(imageB, sampler, (int2)(x, y));
	float4 A;
	float4 min = (float4)(0.0, 0.0, 0.0, 0.0);
	float4 max = (float4)(1.0, 1.0, 1.0, 1.0);
	A.x = output[y * w + x] / 255.0;
	A.y = output[w * h + y * w + x] / 255.0;
	A.z = output[2 * w * h + y * w + x] /255.0;
	A.w = 0.0;
	//if(x < 10 && y < 10)
	//printf("%d,%d: %f\n", x, y, output[y * w + x]);

	write_imagef(imageB, (int2)(x, y), clamp(A, min, max) );
}